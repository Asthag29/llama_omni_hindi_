import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, open_dict
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from omni_speech.datasets.processing.materialize_shard_audio import (
    build_manifest_from_parquets,
    build_sharded_manifests_from_parquets,
    filenames_from_manifest,
    materialize_audio,
)
from omni_speech.trainer import OmniSpeechTrainingModule, SpeechDataModule, SpeechDataset
from omni_speech.utils import (
    build_callbacks,
    build_loggers,
    optional_abs_path,
    resolve_training_state_path,
    save_omni_speech_checkpoint,
)


@dataclass
class ShardPlan:
    train_manifests: List[str]
    train_shard_sample_counts: List[int]
    val_manifest: str
    metadata_path: str
    num_train_shards: int
    total_samples: int
    validation_samples: int
    train_samples: int


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _partition_sizes(total: int, parts: int) -> List[int]:
    if parts <= 0:
        raise ValueError(f"num_train_shards must be positive, got {parts}")
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def _try_load_existing_shard_plan(
    manifest_dir: str,
    num_train_shards: int,
    source_json: str,
) -> Optional[ShardPlan]:
    metadata_path = os.path.join(manifest_dir, "split_meta.json")
    if not os.path.isfile(metadata_path):
        return None

    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)

    if int(meta.get("num_train_shards", -1)) != num_train_shards:
        return None
    if os.path.abspath(str(meta.get("source_json_path", ""))) != os.path.abspath(source_json):
        return None

    val_manifest = meta.get("validation_path")
    if not val_manifest or not os.path.isfile(val_manifest):
        return None

    shards = sorted(meta.get("shards", []), key=lambda item: int(item.get("shard", 0)))
    if len(shards) != num_train_shards:
        return None

    manifest_paths = []
    sample_counts = []
    for expected_idx, shard in enumerate(shards, start=1):
        if int(shard.get("shard", -1)) != expected_idx:
            return None
        path = shard.get("path")
        if not path or not os.path.isfile(path):
            return None
        manifest_paths.append(path)
        sample_counts.append(int(shard.get("sample_count", 0)))

    print(
        f"Reusing existing shard manifests from {manifest_dir} "
        f"({num_train_shards} shards, {sum(sample_counts)} train samples).",
        flush=True,
    )
    return ShardPlan(
        train_manifests=manifest_paths,
        train_shard_sample_counts=sample_counts,
        val_manifest=val_manifest,
        metadata_path=metadata_path,
        num_train_shards=num_train_shards,
        total_samples=int(meta.get("total_samples", 0)),
        validation_samples=int(meta.get("validation_samples", 0)),
        train_samples=int(meta.get("train_samples", sum(sample_counts))),
    )


def prepare_shard_manifests(cfg: DictConfig) -> ShardPlan:
    source_json = optional_abs_path(cfg.sharding.source_json_path)
    manifest_dir = to_absolute_path(str(cfg.sharding.manifest_dir))
    validation_fraction = float(cfg.sharding.validation_fraction)
    num_train_shards = int(cfg.sharding.num_train_shards)
    seed = int(cfg.sharding.seed)
    materialization_cfg = cfg.get("materialization")

    if not source_json or not os.path.isfile(source_json):
        raise FileNotFoundError(f"Shard source manifest not found: {source_json}")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            f"sharding.validation_fraction must be in [0, 1), got {validation_fraction}"
        )

    os.makedirs(manifest_dir, exist_ok=True)
    if bool(cfg.sharding.get("reuse_existing_manifests", True)):
        existing_plan = _try_load_existing_shard_plan(
            manifest_dir,
            num_train_shards,
            source_json,
        )
        if existing_plan is not None:
            return existing_plan

    with open(source_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    total_samples = len(samples)
    if total_samples < 2:
        raise ValueError(
            f"Need at least 2 samples to create fixed validation + train shards, got {total_samples}"
        )

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    val_manifest = os.path.join(manifest_dir, "validation.json")

    validation_source = str(cfg.sharding.get("validation_source", "local_fraction"))
    if validation_source == "hf_split":
        if materialization_cfg is None:
            raise ValueError("sharding.validation_source=hf_split requires materialization config.")
        stats = build_manifest_from_parquets(
            out_path=val_manifest,
            repo_id=str(materialization_cfg.repo_id),
            repo_type=str(materialization_cfg.get("repo_type", "dataset")),
            parquet_prefix=str(materialization_cfg.get("parquet_prefix", "data")),
            parquet_patterns=list(cfg.sharding.validation_parquet_patterns),
            audio_root=to_absolute_path(str(cfg.data.audio_root)),
            index_path=to_absolute_path(str(materialization_cfg.index_path)),
            cache_dir=(
                to_absolute_path(str(materialization_cfg.cache_dir))
                if materialization_cfg.get("cache_dir")
                else None
            ),
            batch_size=int(materialization_cfg.get("batch_size", 512)),
            write_audio=bool(cfg.sharding.get("materialize_validation_audio", True)),
            max_workers=int(materialization_cfg.get("max_workers", 1)),
            progress_every=int(materialization_cfg.get("progress_every", 5)),
            clear_hf_cache_before_download=bool(
                materialization_cfg.get("clear_hf_cache_before_download", False)
            ),
            hf_cache_root=(
                to_absolute_path(str(materialization_cfg.hf_cache_root))
                if materialization_cfg.get("hf_cache_root")
                else None
            ),
        )
        with open(val_manifest, encoding="utf-8") as f:
            val_samples = json.load(f)
        validation_samples = len(val_samples)
        validation_filenames = {
            os.path.basename(str(item.get("speech")))
            for item in val_samples
            if item.get("speech")
        }
        train_samples = [
            item
            for item in shuffled
            if os.path.basename(str(item.get("speech", ""))) not in validation_filenames
        ]
        print(
            "Prepared HF validation split: "
            f"{stats['samples']} samples from {stats['parquet_files']} parquet files "
            f"({stats['audio_written']} audio files written, "
            f"cleared {stats['cleared_hf_cache_bytes'] / (1024 ** 3):.2f} GiB HF cache)."
        )
    else:
        validation_samples = int(round(total_samples * validation_fraction))
        validation_samples = max(1, min(total_samples - 1, validation_samples))
        val_samples = shuffled[:validation_samples]
        train_samples = shuffled[validation_samples:]
        _write_json(val_manifest, val_samples)

    train_source = str(cfg.sharding.get("train_source", "local_manifest"))
    if train_source == "hf_split":
        if materialization_cfg is None:
            raise ValueError("sharding.train_source=hf_split requires materialization config.")
        shard_stats = build_sharded_manifests_from_parquets(
            out_dir=manifest_dir,
            num_shards=num_train_shards,
            repo_id=str(materialization_cfg.repo_id),
            repo_type=str(materialization_cfg.get("repo_type", "dataset")),
            parquet_prefix=str(materialization_cfg.get("parquet_prefix", "data")),
            parquet_patterns=list(cfg.sharding.train_parquet_patterns),
            index_path=to_absolute_path(str(materialization_cfg.index_path)),
            cache_dir=(
                to_absolute_path(str(materialization_cfg.cache_dir))
                if materialization_cfg.get("cache_dir")
                else None
            ),
            batch_size=int(materialization_cfg.get("batch_size", 512)),
            max_workers=int(materialization_cfg.get("max_workers", 1)),
            progress_every=int(materialization_cfg.get("progress_every", 5)),
        )
        manifest_paths = shard_stats["manifest_paths"]
        sample_counts = shard_stats["sample_counts"]
        train_samples_count = shard_stats["samples"]
        shard_ranges = shard_stats.get("shard_ranges", [])
        print(
            "Prepared HF train shards: "
            f"{train_samples_count} samples from {shard_stats['parquet_files']} parquet files.",
            flush=True,
        )
    else:
        shard_sizes = _partition_sizes(len(train_samples), num_train_shards)
        if any(size == 0 for size in shard_sizes):
            raise ValueError(
                f"Some train shards would be empty with {len(train_samples)} train samples "
                f"split across {num_train_shards} shards."
            )
        manifest_paths = []
        sample_counts = []
        start = 0
        for shard_idx, size in enumerate(shard_sizes, start=1):
            shard_manifest = os.path.join(
                manifest_dir,
                f"train_shard_{shard_idx:02d}_of_{num_train_shards:02d}.json",
            )
            shard_samples = train_samples[start : start + size]
            _write_json(shard_manifest, shard_samples)
            manifest_paths.append(shard_manifest)
            sample_counts.append(len(shard_samples))
            start += size
        train_samples_count = len(train_samples)
        shard_ranges = []

    metadata_path = os.path.join(manifest_dir, "split_meta.json")
    _write_json(
        metadata_path,
        {
            "source_json_path": source_json,
            "seed": seed,
            "total_samples": total_samples,
            "validation_fraction": validation_fraction,
            "validation_samples": validation_samples,
            "train_source": train_source,
            "train_samples": train_samples_count,
            "num_train_shards": num_train_shards,
            "shards": [
                {
                    "shard": idx,
                    "path": path,
                    "sample_count": sample_count,
                    **(shard_ranges[idx - 1] if idx - 1 < len(shard_ranges) else {}),
                }
                for idx, (path, sample_count) in enumerate(
                    zip(manifest_paths, sample_counts), start=1
                )
            ],
            "validation_path": val_manifest,
        },
    )

    return ShardPlan(
        train_manifests=manifest_paths,
        train_shard_sample_counts=sample_counts,
        val_manifest=val_manifest,
        metadata_path=metadata_path,
        num_train_shards=num_train_shards,
        total_samples=total_samples,
        validation_samples=validation_samples,
        train_samples=train_samples_count,
    )


def _phase_from_epoch(epoch_idx: int, num_train_shards: int) -> tuple[int, int]:
    return (epoch_idx // num_train_shards) + 1, (epoch_idx % num_train_shards) + 1


def compute_global_optimizer_steps(
    shard_counts: List[int], batch_size: int, grad_accum: int, global_epochs: int
) -> int:
    if batch_size <= 0 or grad_accum <= 0 or global_epochs <= 0:
        raise ValueError("batch_size, grad_accum, and global_epochs must be positive.")
    steps_per_full_epoch = 0
    for sample_count in shard_counts:
        microbatches = math.ceil(sample_count / batch_size)
        steps_per_full_epoch += max(1, math.ceil(microbatches / grad_accum))
    return max(1, steps_per_full_epoch * global_epochs)


def compute_validation_schedule(num_train_shards: int, val_check_interval) -> tuple[int, float]:
    """Map true-epoch validation fractions onto shard-phase validation."""
    if isinstance(val_check_interval, float) and 0.0 < val_check_interval < 1.0:
        return max(1, int(round(num_train_shards * val_check_interval))), 1.0
    return 1, val_check_interval


class CyclingSpeechShardDataModule(SpeechDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer,
        model_config,
        shard_plan: ShardPlan,
    ):
        super().__init__(cfg, tokenizer, model_config)
        self.shard_plan = shard_plan
        self._val_dataset = None
        self._validation_filenames = filenames_from_manifest(shard_plan.val_manifest)
        self._last_train_manifest = None
        self._last_phase = None
        self._last_materialized_key = None

    def _build_dataset(self, manifest_path: str) -> SpeechDataset:
        return SpeechDataset(
            manifest_path,
            self.tokenizer,
            self.model_config,
            input_type=self.cfg.data.input_type,
            mel_size=self.cfg.data.mel_size,
            audio_root=self.cfg.data.get("audio_root"),
        )

    def _materialization_enabled(self) -> bool:
        materialization_cfg = self.cfg.get("materialization")
        return materialization_cfg is not None and bool(materialization_cfg.get("enabled", False))

    def _materialize_for_phase(self, train_manifest_path: Optional[str]) -> None:
        if not self._materialization_enabled():
            return

        train_filenames = filenames_from_manifest(train_manifest_path) if train_manifest_path else set()
        required = self._validation_filenames | train_filenames
        key = (train_manifest_path, len(required))
        if key == self._last_materialized_key:
            return

        materialization_cfg = self.cfg.materialization
        stats = materialize_audio(
            required_filenames=required,
            keep_filenames=required,
            audio_root=to_absolute_path(str(self.cfg.data.audio_root)),
            repo_id=str(materialization_cfg.repo_id),
            repo_type=str(materialization_cfg.get("repo_type", "dataset")),
            parquet_prefix=str(materialization_cfg.get("parquet_prefix", "data")),
            index_path=to_absolute_path(str(materialization_cfg.index_path)),
            cache_dir=(
                to_absolute_path(str(materialization_cfg.cache_dir))
                if materialization_cfg.get("cache_dir")
                else None
            ),
            batch_size=int(materialization_cfg.get("batch_size", 512)),
            delete_extra=bool(materialization_cfg.get("delete_extra", True)),
            max_workers=int(materialization_cfg.get("max_workers", 1)),
            progress_every=int(materialization_cfg.get("progress_every", 5)),
            clear_hf_cache_before_download=bool(
                materialization_cfg.get("clear_hf_cache_before_download", False)
            ),
            hf_cache_root=(
                to_absolute_path(str(materialization_cfg.hf_cache_root))
                if materialization_cfg.get("hf_cache_root")
                else None
            ),
        )
        self._last_materialized_key = key
        print(
            "Materialized shard audio: "
            f"required={stats['required']}, present={stats['present']}, "
            f"downloaded={stats['downloaded']}, deleted={stats['deleted']}, "
            f"indexed={stats['indexed']}, "
            f"cleared_hf_cache={stats['cleared_hf_cache_bytes'] / (1024 ** 3):.2f}GiB"
        )

    def _assert_audio_coverage(self, dataset: SpeechDataset, expected_count: int, label: str) -> None:
        if not bool(self.cfg.sharding.get("strict_audio_coverage", True)):
            return
        actual = len(dataset)
        if actual != expected_count:
            raise RuntimeError(
                f"{label} has {actual}/{expected_count} usable samples. "
                "This sharded workflow expects the current shard audio plus the fixed validation audio "
                "to be present locally before training that phase."
            )

    def current_phase_info(self) -> tuple[int, int]:
        if self.trainer is None:
            return 1, 1
        return _phase_from_epoch(self.trainer.current_epoch, self.shard_plan.num_train_shards)

    def setup(self, stage=None):
        if self._val_dataset is None:
            self._materialize_for_phase(train_manifest_path=None)
            self._val_dataset = self._build_dataset(self.shard_plan.val_manifest)
            self._assert_audio_coverage(
                self._val_dataset,
                self.shard_plan.validation_samples,
                "Fixed validation shard",
            )
        self.val_dataset = self._fraction_subset(
            self._val_dataset,
            float(self.cfg.data.get("val_fraction", 1.0)),
            202,
            "val",
        )
        self.test_dataset = None

    def train_dataloader(self):
        self.setup()
        logical_epoch, shard_idx = self.current_phase_info()
        manifest_path = self.shard_plan.train_manifests[shard_idx - 1]
        expected_count = self.shard_plan.train_shard_sample_counts[shard_idx - 1]
        self._materialize_for_phase(manifest_path)
        train_dataset = self._build_dataset(manifest_path)
        self._assert_audio_coverage(
            train_dataset,
            expected_count,
            f"Train shard {shard_idx}/{self.shard_plan.num_train_shards}",
        )
        self.train_dataset = self._fraction_subset(
            train_dataset,
            float(self.cfg.data.get("train_fraction", 1.0)),
            101,
            "train",
        )
        self._last_train_manifest = manifest_path
        self._last_phase = (logical_epoch, shard_idx)
        print(
            f"Loading train shard {shard_idx}/{self.shard_plan.num_train_shards} "
            f"for logical epoch {logical_epoch}/{int(self.cfg.training.num_train_epochs)}: "
            f"{manifest_path}"
        )
        return super().train_dataloader()

    def val_dataloader(self):
        self.setup()
        return super().val_dataloader()


class GlobalShardTrainingModule(OmniSpeechTrainingModule):
    def __init__(self, cfg: DictConfig, shard_plan: ShardPlan, total_optimizer_steps: int):
        self.shard_plan = shard_plan
        self.total_optimizer_steps = total_optimizer_steps
        super().__init__(cfg)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        logical_epoch, shard_idx = _phase_from_epoch(
            self.current_epoch, self.shard_plan.num_train_shards
        )
        self.log(
            "logical_epoch",
            float(logical_epoch),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "current_train_shard",
            float(shard_idx),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        print(
            f"Starting logical epoch {logical_epoch}/{int(self.cfg.training.num_train_epochs)} "
            f"on shard {shard_idx}/{self.shard_plan.num_train_shards}"
        )

    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        use_8bit = bool(self.cfg.training.get("use_8bit_optimizer", False))
        if use_8bit:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=self.cfg.training.learning_rate,
                weight_decay=self.cfg.training.weight_decay,
            )
            print("Using 8-bit AdamW optimizer.")
        else:
            optimizer = AdamW(
                trainable_params,
                lr=self.cfg.training.learning_rate,
                weight_decay=self.cfg.training.weight_decay,
            )

        if self.cfg.training.lr_scheduler_type != "cosine":
            return optimizer

        warmup_steps = int(
            self.total_optimizer_steps * float(self.cfg.training.warmup_ratio)
        )
        print(
            f"Using global cosine schedule: total_steps={self.total_optimizer_steps}, "
            f"warmup_steps={warmup_steps}"
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_optimizer_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def _resolve_optional_resume_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return resolve_training_state_path(optional_abs_path(path) or path)


def _write_shard_run_metadata(
    cfg: DictConfig,
    shard_plan: ShardPlan,
    resume_path: Optional[str],
    total_optimizer_steps: int,
) -> None:
    metadata_path = to_absolute_path(os.path.join(str(cfg.logging.output_dir), "shard_run.json"))
    _write_json(
        metadata_path,
        {
            "num_train_shards": shard_plan.num_train_shards,
            "global_num_train_epochs": int(cfg.training.num_train_epochs),
            "global_total_phases": int(cfg.training.num_train_epochs)
            * shard_plan.num_train_shards,
            "global_total_optimizer_steps": total_optimizer_steps,
            "check_val_every_n_shard_phases": int(
                cfg.training.check_val_every_n_shard_phases
            ),
            "total_samples": shard_plan.total_samples,
            "validation_samples": shard_plan.validation_samples,
            "train_samples": shard_plan.train_samples,
            "train_shard_sample_counts": shard_plan.train_shard_sample_counts,
            "train_manifests": shard_plan.train_manifests,
            "validation_manifest": shard_plan.val_manifest,
            "split_metadata": shard_plan.metadata_path,
            "resume_path": resume_path,
        },
    )


@hydra.main(version_base=None, config_path="../configs", config_name="speech_shard")
def main(cfg: DictConfig):
    pl.seed_everything(int(cfg.data.seed), workers=True)

    shard_plan = prepare_shard_manifests(cfg)
    total_optimizer_steps = compute_global_optimizer_steps(
        shard_plan.train_shard_sample_counts,
        int(cfg.training.batch_size),
        int(cfg.training.gradient_accumulation_steps),
        int(cfg.training.num_train_epochs),
    )
    global_total_phases = int(cfg.training.num_train_epochs) * shard_plan.num_train_shards
    check_val_every_n_epoch, val_check_interval = compute_validation_schedule(
        shard_plan.num_train_shards,
        float(cfg.training.val_check_interval),
    )

    with open_dict(cfg):
        cfg.data.json_path = shard_plan.train_manifests[0]
        cfg.data.validation_split = 0.0
        cfg.data.test_split = 0.0
        cfg.data.current_validation_manifest = shard_plan.val_manifest
        cfg.training.global_total_phases = global_total_phases
        cfg.training.global_total_optimizer_steps = total_optimizer_steps
        cfg.training.check_val_every_n_shard_phases = check_val_every_n_epoch

    resume_cfg = cfg.get("resume")
    resume_path = resume_cfg.get("path") if resume_cfg is not None else None
    resolved_resume_path = _resolve_optional_resume_path(resume_path)

    _write_shard_run_metadata(cfg, shard_plan, resolved_resume_path, total_optimizer_steps)

    module = GlobalShardTrainingModule(cfg, shard_plan, total_optimizer_steps)
    data_module = CyclingSpeechShardDataModule(
        cfg,
        module.tokenizer,
        module.model.config,
        shard_plan=shard_plan,
    )
    data_module.setup()
    has_validation = data_module.val_dataset is not None

    trainer = pl.Trainer(
        default_root_dir=to_absolute_path(str(cfg.logging.output_dir)),
        max_epochs=global_total_phases,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.strategy,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        gradient_clip_val=cfg.training.max_grad_norm,
        logger=build_loggers(cfg),
        callbacks=build_callbacks(cfg, has_validation),
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        val_check_interval=val_check_interval,
        fast_dev_run=cfg.training.fast_dev_run,
        enable_checkpointing=False,
        reload_dataloaders_every_n_epochs=1,
    )

    if resolved_resume_path:
        print(f"Resuming shard trainer state from {resolved_resume_path}")

    trainer.fit(module, datamodule=data_module, ckpt_path=resolved_resume_path)
    final_dir = to_absolute_path(os.path.join(cfg.logging.output_dir, "final_model"))
    save_omni_speech_checkpoint(
        module,
        final_dir,
        metadata={
            "final": True,
            "num_train_shards": shard_plan.num_train_shards,
            "global_num_train_epochs": int(cfg.training.num_train_epochs),
            "global_total_optimizer_steps": total_optimizer_steps,
        },
    )
    if trainer.global_rank == 0:
        module.tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
