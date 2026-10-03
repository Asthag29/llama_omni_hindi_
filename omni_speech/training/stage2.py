"""Stage-2 OmniSpeech training on the local parquet speech data.

``data.speech_dir`` holds ``{train,validation,test}/*.parquet`` plus a
``manifest.json``, written by
``python -m omni_speech.datasets.processing.build_stage2_local``. Row counts for
the LR schedule come from the manifest. Rows are read with the ``datasets``
library's iterable parquet reader; audio bytes are decoded in the DataLoader
workers.
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import hydra
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import torchaudio
import whisper
# Imported here, in the main process, on purpose. W&B hooks the first import of
# `datasets`; inside a forked DataLoader worker that hook waits on a W&B thread
# that does not exist there, and the worker deadlocks before yielding a row.
from datasets import Audio, load_dataset
from datasets.distributed import split_dataset_by_node
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from omni_speech.constants import DEFAULT_SPEECH_PROMPT
from omni_speech.datasets.preprocess import preprocess, preprocess_multimodal
from omni_speech.training.speech_module import OmniSpeechTrainingModule, SpeechCollator
from omni_speech.train_utils import (
    build_callbacks,
    build_loggers,
    finalize_fit_outputs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_LOCAL_COMMAND = "python -m omni_speech.datasets.processing.build_stage2_local"
LOCAL_SPLITS = ("train", "validation", "test")
# Whisper's encoder sees a fixed 30 s window (N_SAMPLES at 16 kHz); longer
# clips would be silently cut by pad_or_trim, so they are skipped instead.
MAX_AUDIO_SAMPLES = int(whisper.audio.N_SAMPLES)
# Upper bound on parquet files interleaved into one shuffle buffer
# (datasets' own default for IterableDataset.shuffle).
MAX_INTERLEAVED_FILES = 10


class LocalDataError(RuntimeError):
    """The local stage-2 data directory is missing, incomplete or inconsistent."""


def resolve_repo_path(path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(str(path)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_local_manifest(data_dir: Path) -> dict:
    """Read ``<data_dir>/manifest.json`` and check it describes a finished build."""
    build_hint = f"Build it with: {BUILD_LOCAL_COMMAND}"
    if not data_dir.is_dir():
        raise LocalDataError(f"Stage-2 data directory not found: {data_dir}. {build_hint}")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise LocalDataError(f"Stage-2 manifest not found: {manifest_path}. {build_hint}")
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("complete") is not True:
        raise LocalDataError(
            f"Stage-2 data in {data_dir} is incomplete (manifest 'complete' is "
            f"{manifest.get('complete')!r}). Finish the build with: {BUILD_LOCAL_COMMAND}"
        )
    splits = manifest.get("splits") or {}
    for split in LOCAL_SPLITS:
        if split not in splits or "rows" not in splits[split]:
            raise LocalDataError(
                f"Stage-2 manifest {manifest_path} has no row count for split '{split}'. {build_hint}"
            )
    return manifest


def list_local_split_files(data_dir: Path, split: str, manifest: dict) -> list[str]:
    """Parquet files of one local split; their number must match the manifest."""
    files = sorted(str(path) for path in (data_dir / split).glob("*.parquet"))
    if not files:
        raise LocalDataError(
            f"No parquet files in {data_dir / split}. Build the data with: {BUILD_LOCAL_COMMAND}"
        )
    expected = manifest["splits"][split].get("files")
    if expected is not None and int(expected) != len(files):
        raise LocalDataError(
            f"{data_dir / split} has {len(files)} parquet files but the manifest lists "
            f"{expected}. Rebuild the data with: {BUILD_LOCAL_COMMAND}"
        )
    return files


@dataclass
class Stage2DataSource:
    """The local stage-2 data directory and its row counts per split."""

    data_dir: Path
    manifest: dict
    train_samples: int
    validation_samples: int
    test_samples: int
    split_files: dict[str, list[str]]

    def summary(self) -> dict:
        return {
            "speech_dir": str(self.data_dir),
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
            "test_samples": self.test_samples,
            "manifest_splits": self.manifest.get("splits"),
            "manifest_max_audio_seconds": self.manifest.get("max_audio_seconds"),
        }


def resolve_data_source(data_cfg: DictConfig) -> Stage2DataSource:
    """Validate ``data.speech_dir`` and read its row counts from the manifest."""
    speech_dir = data_cfg.get("speech_dir")
    if speech_dir in (None, ""):
        raise LocalDataError(
            f"data.speech_dir is not set. Point it at the stage-2 data directory "
            f"(default data/speech), built with: {BUILD_LOCAL_COMMAND}"
        )
    data_dir = resolve_repo_path(speech_dir)
    manifest = load_local_manifest(data_dir)
    splits = manifest["splits"]
    return Stage2DataSource(
        train_samples=int(splits["train"]["rows"]),
        validation_samples=int(splits["validation"]["rows"]),
        test_samples=int(splits["test"]["rows"]),
        data_dir=data_dir,
        manifest=manifest,
        split_files={
            split: list_local_split_files(data_dir, split, manifest)
            for split in ("train", "validation")
        },
    )


def _source_rows(value) -> int | None:
    if isinstance(value, dict):
        value = value.get("rows")
    return int(value) if value is not None else None


def format_split_table(source: Stage2DataSource) -> str:
    """Rows per split and per source for the startup log."""
    splits = source.manifest["splits"]
    lines = [
        f"Stage-2 data: {source.data_dir} "
        f"(max_audio_seconds={source.manifest.get('max_audio_seconds')})",
        f"  {'split':<12}{'rows':>10}{'files':>8}{'hours':>10}",
    ]
    for split in LOCAL_SPLITS:
        info = splits[split]
        hours = info.get("hours")
        hours_text = f"{float(hours):.2f}" if hours is not None else "-"
        lines.append(
            f"  {split:<12}{int(info['rows']):>10}{str(info.get('files', '-')):>8}{hours_text:>10}"
        )

    sources = sorted({name for split in LOCAL_SPLITS for name in (splits[split].get("by_source") or {})})
    if sources:
        width = max(12, max(len(name) for name in sources) + 2)
        lines.append("  rows per source:")
        lines.append(f"  {'source':<{width}}" + "".join(f"{split:>12}" for split in LOCAL_SPLITS))
        for name in sources:
            cells = []
            for split in LOCAL_SPLITS:
                rows = _source_rows((splits[split].get("by_source") or {}).get(name))
                cells.append(f"{rows if rows is not None else '-':>12}")
            lines.append(f"  {name:<{width}}" + "".join(cells))
    return "\n".join(lines)


def _buffer_input_shards(num_files: int, num_consumers: int) -> int:
    """How many files to interleave into one shuffle buffer.

    ``IterableDataset.shuffle`` interleaves ``k`` groups of files; the shuffled
    dataset then exposes ``num_files // k`` shards, and ``datasets`` hands
    those shards out to ranks and DataLoader workers. Keeping
    ``num_files // k >= num_consumers`` (ranks x workers) keeps every worker
    busy; ``k == 1`` disables interleaving so each file is its own shard.
    """
    return max(1, min(MAX_INTERLEAVED_FILES, num_files // max(1, num_consumers)))


def _load_parquet_dataset(data_files: list[str], split_name: str):
    """Iterable (``streaming=True``) reader over local parquet files; writes no cache."""

    dataset = load_dataset(
        "parquet",
        data_files={split_name: data_files},
        split=split_name,
        streaming=True,
    )
    # Keep raw audio bytes/path in rows. If HF Datasets decodes Audio itself, it
    # requires torchcodec in recent versions; our loader already decodes bytes.
    return dataset.cast_column("audio", Audio(decode=False))


def _resample_if_needed(audio: np.ndarray, source_rate: int, target_rate: int = 16000) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if source_rate == target_rate:
        return audio.astype(np.float32)

    waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    waveform = torchaudio.functional.resample(waveform, source_rate, target_rate)
    return waveform.squeeze(0).numpy().astype(np.float32)


def _decode_audio_value(audio_value) -> np.ndarray | None:
    if audio_value is None:
        return None

    if isinstance(audio_value, dict):
        if audio_value.get("array") is not None:
            sampling_rate = int(audio_value.get("sampling_rate") or 16000)
            return _resample_if_needed(np.asarray(audio_value["array"], dtype=np.float32), sampling_rate)

        audio_bytes = audio_value.get("bytes")
        if audio_bytes is not None:
            audio, sampling_rate = sf.read(io.BytesIO(bytes(audio_bytes)), dtype="float32", always_2d=False)
            return _resample_if_needed(audio, int(sampling_rate))

        audio_path = audio_value.get("path")
        if audio_path and os.path.exists(str(audio_path)):
            audio, sampling_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
            return _resample_if_needed(audio, int(sampling_rate))
        return None

    if isinstance(audio_value, (bytes, bytearray, memoryview)):
        audio, sampling_rate = sf.read(io.BytesIO(bytes(audio_value)), dtype="float32", always_2d=False)
        return _resample_if_needed(audio, int(sampling_rate))

    return None


def _row_to_conversations(row: dict) -> list[dict]:
    conversations = row.get("conversations")
    if conversations:
        return copy.deepcopy(conversations)

    assistant_text = (row.get("assistant_text") or row.get("text") or "").strip()
    return [
        {"from": "human", "value": DEFAULT_SPEECH_PROMPT},
        {"from": "gpt", "value": assistant_text},
    ]


class Stage2SpeechDataset(IterableDataset):
    """Iterates parquet rows and turns them into stage-2 training items.

    Partitioning: rows are never filtered by index here. ``datasets`` assigns
    disjoint parquet files (shards) to each DataLoader worker
    (``IterableDataset._iter_pytorch``) and, when ``world_size > 1``, to each
    rank (``split_dataset_by_node``: whole shards when the shard count divides
    by ``world_size``, otherwise every ``world_size``-th row of the identical
    per-worker stream). Every row of one pass is therefore read by exactly one
    (rank, worker). Workers beyond the shard count receive nothing.
    """

    def __init__(
        self,
        data_files: list[str],
        tokenizer,
        split_name: str,
        seed: int,
        shuffle_buffer_size: int,
        repeat: bool,
        mel_size: int = 128,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.split_name = split_name
        self.seed = int(seed)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.repeat = repeat
        self.mel_size = int(mel_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"Invalid rank {self.rank} for world_size {self.world_size}")
        self._skipped_overlength = 0
        self._skipped_overlong_audio = 0
        self._rows_in_pass = 0
        self.data_args = type("DataArgs", (), {"is_multimodal": True})()

    def _log_skip(self, reason: str, count: int, row: dict, detail: str) -> None:
        if count <= 10 or count % 100 == 0:
            worker = get_worker_info()
            print(
                f"Skipped {reason} sample "
                f"split={self.split_name} id={row.get('id', '<unknown>')} "
                f"rank={self.rank} worker={worker.id if worker is not None else 0} "
                f"{detail} skipped_{reason.replace('-', '_')}={count}",
                flush=True,
            )

    def _decode_row_audio(self, row: dict) -> np.ndarray | None:
        """16 kHz mono audio of the row, or None if missing or longer than 30 s."""
        audio = _decode_audio_value(row.get("audio"))
        if audio is None:
            return None
        if audio.shape[0] > MAX_AUDIO_SAMPLES:
            self._skipped_overlong_audio += 1
            self._log_skip(
                "overlong-audio",
                self._skipped_overlong_audio,
                row,
                f"seconds={audio.shape[0] / 16000:.2f} max={MAX_AUDIO_SAMPLES / 16000:.0f}",
            )
            return None
        return audio

    def _row_to_item(self, row: dict) -> dict | None:
        audio = self._decode_row_audio(row)
        if audio is None:
            return None

        conversations = _row_to_conversations(row)
        source = preprocess_multimodal([conversations], self.data_args)[0]
        text = preprocess([source], self.tokenizer, has_speech=True)
        token_length = int(text["input_ids"].shape[-1])
        max_length = int(self.tokenizer.model_max_length)
        if token_length > max_length:
            self._skipped_overlength += 1
            self._log_skip(
                "overlength",
                self._skipped_overlength,
                row,
                f"tokens={token_length} max={max_length}",
            )
            return None

        # Log-mel features (frames, n_mels) of the 30 s Whisper window.
        audio = whisper.pad_or_trim(audio)
        speech = whisper.log_mel_spectrogram(audio, n_mels=self.mel_size).permute(1, 0)

        return {
            "input_ids": text["input_ids"].squeeze(0),
            "labels": text["labels"].squeeze(0),
            "speech": speech,
            "speech_length": torch.tensor(speech.shape[0], dtype=torch.long),
        }

    def _iter_rows(self, pass_idx: int) -> Iterator[dict]:
        """Raw parquet rows of one pass that belong to this rank and worker."""
        dataset = _load_parquet_dataset(self.data_files, self.split_name)
        if self.shuffle_buffer_size > 0:
            worker = get_worker_info()
            num_workers = worker.num_workers if worker is not None else 1
            # Same seed in every rank and worker: they all derive the same file
            # order and then take disjoint shards of it.
            dataset = dataset.shuffle(
                seed=self.seed + pass_idx,
                buffer_size=self.shuffle_buffer_size,
                max_buffer_input_shards=_buffer_input_shards(
                    len(self.data_files), num_workers * self.world_size
                ),
            )
        if self.world_size > 1:
            dataset = split_dataset_by_node(dataset, rank=self.rank, world_size=self.world_size)
        yield from dataset

    def _iter_one_pass(self, pass_idx: int) -> Iterator[dict]:
        for row in self._iter_rows(pass_idx):
            self._rows_in_pass += 1
            item = self._row_to_item(row)
            if item is not None:
                yield item

    def __iter__(self) -> Iterator[dict]:
        pass_idx = 0
        while True:
            self._rows_in_pass = 0
            yield from self._iter_one_pass(pass_idx)
            if not self.repeat or self._rows_in_pass == 0:
                # A worker that was assigned no files (fewer files than
                # workers) ends here instead of spinning forever on empty
                # passes, which would stall the DataLoader.
                break
            pass_idx += 1


class Stage2SpeechDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer,
        data_source: Stage2DataSource | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.data_source = data_source or resolve_data_source(cfg.data)
        self.train_files: list[str] = list(self.data_source.split_files["train"])
        self.val_files: list[str] = list(self.data_source.split_files["validation"])
        self.test_files: list[str] = []

    def _loader_kwargs(self) -> dict:
        num_workers = int(self.cfg.data.num_workers)
        return {
            "num_workers": num_workers,
            "collate_fn": SpeechCollator(self.tokenizer),
            "pin_memory": torch.cuda.is_available(),
            # Keep the validation workers alive between validation runs instead
            # of re-spawning them (and re-pickling the tokenizer) every time.
            "persistent_workers": num_workers > 0,
        }

    def _rank_and_world_size(self) -> tuple[int, int]:
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            return int(trainer.global_rank), int(trainer.world_size)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _build_dataset(
        self,
        files: list[str],
        split_name: str,
        seed: int,
        shuffle_buffer_size: int,
        repeat: bool,
    ) -> Stage2SpeechDataset:
        rank, world_size = self._rank_and_world_size()
        return Stage2SpeechDataset(
            data_files=files,
            tokenizer=self.tokenizer,
            split_name=split_name,
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
            repeat=repeat,
            mel_size=int(self.cfg.data.mel_size),
            rank=rank,
            world_size=world_size,
        )

    def _eval_dataloader(self, files: list[str], split_name: str) -> DataLoader:
        # Not shuffled: every row once, in the same order on every run.
        dataset = self._build_dataset(
            files,
            split_name,
            seed=int(self.cfg.data.seed) + 10_000,
            shuffle_buffer_size=0,
            repeat=False,
        )
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.training.batch_size),
            **self._loader_kwargs(),
        )

    def train_dataloader(self):
        dataset = self._build_dataset(
            self.train_files,
            "train",
            seed=int(self.cfg.data.seed),
            shuffle_buffer_size=int(self.cfg.data.shuffle_buffer_size),
            repeat=True,
        )
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.training.batch_size),
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return self._eval_dataloader(self.val_files, "validation")

    def test_dataloader(self):
        """Held-out test split; never used during ``fit``."""
        source = self.data_source
        if not self.test_files:
            self.test_files = list_local_split_files(source.data_dir, "test", source.manifest)
        return self._eval_dataloader(self.test_files, "test")


def compute_total_optimizer_steps(train_samples: int, batch_size: int, grad_accum: int, epochs: int) -> int:
    microbatches_per_epoch = math.ceil(train_samples / batch_size)
    return max(1, math.ceil(microbatches_per_epoch / grad_accum) * epochs)


def compute_schedule(
    train_samples: int,
    batch_size: int,
    grad_accum: int,
    epochs: int,
    warmup_ratio: float,
) -> dict:
    """Optimizer-step schedule for one process seeing ``train_samples`` rows per epoch."""
    microbatches_per_epoch = math.ceil(train_samples / batch_size)
    total_steps = compute_total_optimizer_steps(train_samples, batch_size, grad_accum, epochs)
    return {
        "train_samples": int(train_samples),
        "microbatches_per_epoch": microbatches_per_epoch,
        "optimizer_steps_per_epoch": math.ceil(microbatches_per_epoch / grad_accum),
        "total_optimizer_steps": total_steps,
        # Same formula as OmniSpeechTrainingModule.configure_optimizers.
        "warmup_steps": int(total_steps * float(warmup_ratio)),
    }


def compute_validation_interval_batches(train_samples: int, batch_size: int, val_check_interval) -> int | float:
    if isinstance(val_check_interval, float) and 0.0 < val_check_interval < 1.0:
        microbatches_per_epoch = math.ceil(train_samples / batch_size)
        return max(1, int(round(microbatches_per_epoch * val_check_interval)))
    return val_check_interval


def plan_schedule(cfg: DictConfig, data_source: Stage2DataSource) -> tuple[dict, int | float]:
    """Compute the step schedule from the data source counts and print it."""
    train_samples = data_source.train_samples
    schedule = compute_schedule(
        train_samples,
        int(cfg.training.batch_size),
        int(cfg.training.gradient_accumulation_steps),
        int(cfg.training.num_train_epochs),
        float(cfg.training.warmup_ratio),
    )
    val_interval = compute_validation_interval_batches(
        train_samples,
        int(cfg.training.batch_size),
        cfg.training.val_check_interval,
    )
    print(format_split_table(data_source), flush=True)
    print(
        f"Stage-2 schedule: train_rows={train_samples} "
        f"batch_size={int(cfg.training.batch_size)} "
        f"grad_accum={int(cfg.training.gradient_accumulation_steps)} "
        f"epochs={int(cfg.training.num_train_epochs)} -> "
        f"optimizer_steps_per_epoch={schedule['optimizer_steps_per_epoch']} "
        f"total_steps={schedule['total_optimizer_steps']} "
        f"warmup_steps={schedule['warmup_steps']} "
        f"(warmup_ratio={float(cfg.training.warmup_ratio)}), "
        f"val_check_interval_batches={val_interval}",
        flush=True,
    )
    return schedule, val_interval


class Stage2TrainingModule(OmniSpeechTrainingModule):
    """Speech module trained for a fixed number of optimizer steps.

    The training stream repeats forever (Lightning never sees an epoch end), so
    the cosine schedule length comes from the manifest row count instead of
    ``trainer.estimated_stepping_batches``.
    """

    def __init__(self, cfg: DictConfig, total_optimizer_steps: int, train_samples: int):
        self.total_optimizer_steps = total_optimizer_steps
        self.microbatches_per_epoch = math.ceil(
            int(train_samples) / int(cfg.training.batch_size)
        )
        self.optimizer_steps_per_epoch = math.ceil(
            self.microbatches_per_epoch / int(cfg.training.gradient_accumulation_steps)
        )
        super().__init__(cfg)

    def _log_epoch_progress(self, batch_idx: int) -> None:
        completed_microbatches = int(self.global_step) * int(
            self.cfg.training.gradient_accumulation_steps
        ) + int(batch_idx) % int(self.cfg.training.gradient_accumulation_steps)
        true_epoch = completed_microbatches / max(1, self.microbatches_per_epoch)
        self.log(
            "true_epoch",
            true_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "microbatch_progress",
            float(completed_microbatches),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "optimizer_epoch",
            float(self.global_step) / max(1, self.optimizer_steps_per_epoch),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        self._log_epoch_progress(batch_idx)
        return super().training_step(batch, batch_idx)

    def _total_optimizer_steps(self) -> int:
        return self.total_optimizer_steps


@hydra.main(version_base=None, config_path="../../configs", config_name="stage_2")
def main(cfg: DictConfig):
    pl.seed_everything(int(cfg.data.seed), workers=True)

    # Resolve (and validate) the data before loading the model, so a missing
    # local build fails fast.
    data_source = resolve_data_source(cfg.data)
    train_samples = data_source.train_samples
    schedule, val_interval = plan_schedule(cfg, data_source)
    total_optimizer_steps = schedule["total_optimizer_steps"]

    run_summary = {
        **data_source.summary(),
        "global_num_train_epochs": int(cfg.training.num_train_epochs),
        "optimizer_steps_per_epoch": schedule["optimizer_steps_per_epoch"],
        "global_total_optimizer_steps": total_optimizer_steps,
        "warmup_steps": schedule["warmup_steps"],
        "val_check_interval_batches": val_interval,
    }
    output_dir = to_absolute_path(str(cfg.logging.output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "stage2_run.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    module = Stage2TrainingModule(cfg, total_optimizer_steps, train_samples)
    data_module = Stage2SpeechDataModule(cfg, module.tokenizer, data_source=data_source)
    has_validation = bool(data_module.val_files)

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=-1,
        max_steps=total_optimizer_steps,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.strategy,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        gradient_clip_val=cfg.training.max_grad_norm,
        logger=build_loggers(cfg),
        callbacks=build_callbacks(cfg, has_validation),
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=val_interval,
        fast_dev_run=cfg.training.fast_dev_run,
        enable_checkpointing=False,
    )

    trainer.fit(module, datamodule=data_module)
    finalize_fit_outputs(
        trainer, module, output_dir, final_metadata={"stage": 2}, tokenizer=module.tokenizer,
    )


if __name__ == "__main__":
    main()
