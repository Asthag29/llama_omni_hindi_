"""Stage-2 OmniSpeech training over parquet speech data.

Two data modes, selected by ``streaming.data_dir``:

* local (default): ``<data_dir>/{train,validation,test}/*.parquet`` plus a
  ``manifest.json`` written by
  ``python -m omni_speech.datasets.processing.build_stage2_local``. Sample
  counts for the schedule come from the manifest.
* Hub (``streaming.data_dir: null``): parquet files streamed from
  ``streaming.repo_id`` using the ``*_parquet_patterns`` and the
  ``train_samples``/``validation_samples`` counts in the config.

Both modes read rows with ``datasets`` in streaming mode; audio bytes are
decoded from the parquet rows in the dataloader workers.
"""

from __future__ import annotations

import copy
import fnmatch
import io
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import hydra
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import torchaudio
import whisper
from huggingface_hub import HfApi
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import get_cosine_schedule_with_warmup

from omni_speech.constants import DEFAULT_SPEECH_PROMPT
from omni_speech.datasets.preprocess import preprocess, preprocess_multimodal
from omni_speech.training.combined import OmniSpeechTrainingModule, SpeechCollator
from omni_speech.train_utils import (
    build_callbacks,
    build_loggers,
    save_omni_speech_checkpoint,
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


def list_matching_parquet_files(repo_id: str, repo_type: str, parquet_prefix: str, patterns: Iterable[str]) -> list[str]:
    repo_files = HfApi().list_repo_files(repo_id=repo_id, repo_type=repo_type)
    prefix = parquet_prefix.strip("/")
    matches: list[str] = []
    for pattern in patterns:
        full_pattern = f"{prefix}/{pattern}" if prefix and not str(pattern).startswith(f"{prefix}/") else str(pattern)
        matches.extend(fnmatch.filter(repo_files, full_pattern))
    return sorted(set(matches))


def _hf_data_url(repo_id: str, repo_type: str, path: str) -> str:
    if repo_type != "dataset":
        raise ValueError("HF parquet streaming currently expects repo_type='dataset'.")
    return f"hf://datasets/{repo_id}/{path}"


def _resolve_hf_data_files(cfg: DictConfig, patterns: Iterable[str]) -> list[str]:
    repo_id = str(cfg.repo_id)
    repo_type = str(cfg.get("repo_type", "dataset"))
    parquet_prefix = str(cfg.get("parquet_prefix", "data"))
    matches = list_matching_parquet_files(repo_id, repo_type, parquet_prefix, patterns)
    if not matches:
        raise RuntimeError(
            f"No parquet files matched patterns {list(patterns)} in {repo_id}/{parquet_prefix}"
        )
    return [_hf_data_url(repo_id, repo_type, path) for path in matches]


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
    """Where stage-2 rows come from and how many there are per split."""

    mode: str  # "local" or "hub"
    train_samples: int
    validation_samples: int | None
    test_samples: int | None = None
    data_dir: Path | None = None
    manifest: dict | None = None
    split_files: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> dict:
        summary = {
            "mode": self.mode,
            "data_dir": str(self.data_dir) if self.data_dir is not None else None,
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
            "test_samples": self.test_samples,
        }
        if self.manifest is not None:
            summary["manifest_splits"] = self.manifest.get("splits")
            summary["manifest_max_audio_seconds"] = self.manifest.get("max_audio_seconds")
        return summary


def resolve_data_source(streaming_cfg: DictConfig) -> Stage2DataSource:
    """Local mode when ``streaming.data_dir`` is set, otherwise Hub mode."""
    data_dir = streaming_cfg.get("data_dir")
    if data_dir in (None, ""):
        validation_samples = streaming_cfg.get("validation_samples")
        return Stage2DataSource(
            mode="hub",
            train_samples=int(streaming_cfg.train_samples),
            validation_samples=int(validation_samples) if validation_samples is not None else None,
        )

    data_dir = resolve_repo_path(data_dir)
    manifest = load_local_manifest(data_dir)
    splits = manifest["splits"]
    return Stage2DataSource(
        mode="local",
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
    """Rows per split (and per source in local mode) for the startup log."""
    if source.mode == "hub":
        return (
            "Stage-2 data: Hub mode (counts from config)\n"
            f"  train rows       {source.train_samples}\n"
            f"  validation rows  {source.validation_samples}"
        )

    splits = source.manifest["splits"]
    lines = [
        f"Stage-2 data: local {source.data_dir} "
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


def _load_streaming_dataset(data_files: list[str], split_name: str, cache_dir: str | None):
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise ImportError(
            "HF streaming training requires the `datasets` package. "
            "Install it in the environment before running stage2.py."
        ) from exc

    dataset = load_dataset(
        "parquet",
        data_files={split_name: data_files},
        split=split_name,
        streaming=True,
        cache_dir=cache_dir,
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


class HFStreamingSpeechDataset(IterableDataset):
    """Streams parquet rows and turns them into stage-2 training items.

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
        model_config,
        split_name: str,
        cache_dir: str | None,
        seed: int,
        shuffle_buffer_size: int,
        repeat: bool,
        input_type: str = "mel",
        mel_size: int = 128,
        compute_mel_on_gpu: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.split_name = split_name
        self.cache_dir = cache_dir
        self.seed = int(seed)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.repeat = repeat
        self.input_type = input_type
        self.mel_size = int(mel_size)
        self.compute_mel_on_gpu = bool(compute_mel_on_gpu)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"Invalid rank {self.rank} for world_size {self.world_size}")
        self._skipped_overlength = 0
        self._skipped_overlong_audio = 0
        self._rows_in_pass = 0
        self.data_args = type(
            "DataArgs",
            (),
            {"is_multimodal": True, "input_type": input_type, "mel_size": self.mel_size},
        )()

    def _log_skip(self, reason: str, count: int, row: dict, detail: str) -> None:
        if count <= 10 or count % 100 == 0:
            worker = get_worker_info()
            print(
                f"Skipped {reason} streaming sample "
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

        if self.input_type == "raw" or (self.input_type == "mel" and self.compute_mel_on_gpu):
            speech = torch.from_numpy(audio)
            if getattr(self.model_config, "speech_normalize", False):
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
            speech_length = speech.shape[0]
        elif self.input_type == "mel":
            audio = whisper.pad_or_trim(audio)
            speech = whisper.log_mel_spectrogram(audio, n_mels=self.mel_size).permute(1, 0)
            speech_length = speech.shape[0]
        else:
            raise ValueError(f"Unsupported input_type: {self.input_type}")

        return {
            "input_ids": text["input_ids"].squeeze(0),
            "labels": text["labels"].squeeze(0),
            "speech": speech,
            "speech_length": torch.tensor(speech_length, dtype=torch.long),
        }

    def _iter_rows(self, pass_idx: int) -> Iterator[dict]:
        """Raw parquet rows of one pass that belong to this rank and worker."""
        dataset = _load_streaming_dataset(self.data_files, self.split_name, self.cache_dir)
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
            from datasets.distributed import split_dataset_by_node

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


class HFStreamingSpeechDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer,
        model_config,
        data_source: Stage2DataSource | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.data_source = data_source or resolve_data_source(cfg.streaming)
        self.train_files: list[str] = list(self.data_source.split_files.get("train", []))
        self.val_files: list[str] = list(self.data_source.split_files.get("validation", []))
        self.test_files: list[str] = []

    def _loader_kwargs(self) -> dict:
        num_workers = int(self.cfg.data.num_workers)
        kwargs = {
            "num_workers": num_workers,
            "collate_fn": SpeechCollator(self.tokenizer),
            "pin_memory": torch.cuda.is_available(),
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(self.cfg.data.get("prefetch_factor", 2))
            kwargs["persistent_workers"] = bool(
                self.cfg.data.get("persistent_workers", False)
            )
        return kwargs

    def _rank_and_world_size(self) -> tuple[int, int]:
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            return int(trainer.global_rank), int(trainer.world_size)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def setup(self, stage=None):
        if self.data_source.mode == "local":
            return
        streaming_cfg = self.cfg.streaming
        if not self.train_files:
            self.train_files = _resolve_hf_data_files(
                streaming_cfg,
                streaming_cfg.train_parquet_patterns,
            )
            print(f"HF streaming train files: {len(self.train_files)} parquet files")
        if not self.val_files and streaming_cfg.get("validation_parquet_patterns"):
            self.val_files = _resolve_hf_data_files(
                streaming_cfg,
                streaming_cfg.validation_parquet_patterns,
            )
            print(f"HF streaming validation files: {len(self.val_files)} parquet files")

    def _build_dataset(
        self,
        files: list[str],
        split_name: str,
        seed: int,
        shuffle_buffer_size: int,
        repeat: bool,
    ) -> HFStreamingSpeechDataset:
        rank, world_size = self._rank_and_world_size()
        return HFStreamingSpeechDataset(
            data_files=files,
            tokenizer=self.tokenizer,
            model_config=self.model_config,
            split_name=split_name,
            cache_dir=self.cfg.streaming.get("cache_dir"),
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
            repeat=repeat,
            input_type=self.cfg.data.input_type,
            mel_size=int(self.cfg.data.mel_size),
            compute_mel_on_gpu=bool(self.cfg.data.get("compute_mel_on_gpu", False)),
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
        self.setup()
        dataset = self._build_dataset(
            self.train_files,
            "train",
            seed=int(self.cfg.data.seed),
            shuffle_buffer_size=int(self.cfg.streaming.shuffle_buffer_size),
            repeat=True,
        )
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.training.batch_size),
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        self.setup()
        if not self.val_files:
            return None
        return self._eval_dataloader(self.val_files, "validation")

    def test_dataloader(self):
        """Test split (local mode only); never used during ``fit``."""
        source = self.data_source
        if source.mode != "local":
            raise RuntimeError("The stage-2 test split is only available in local mode (streaming.data_dir).")
        if not self.test_files:
            self.test_files = list_local_split_files(source.data_dir, "test", source.manifest)
        return self._eval_dataloader(self.test_files, "test")


def compute_streaming_optimizer_steps(train_samples: int, batch_size: int, grad_accum: int, epochs: int) -> int:
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
    total_steps = compute_streaming_optimizer_steps(train_samples, batch_size, grad_accum, epochs)
    return {
        "train_samples": int(train_samples),
        "microbatches_per_epoch": microbatches_per_epoch,
        "optimizer_steps_per_epoch": math.ceil(microbatches_per_epoch / grad_accum),
        "total_optimizer_steps": total_steps,
        # Same formula as HFStreamingTrainingModule.configure_optimizers.
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


class HFStreamingTrainingModule(OmniSpeechTrainingModule):
    def __init__(self, cfg: DictConfig, total_optimizer_steps: int, train_samples: int):
        self.total_optimizer_steps = total_optimizer_steps
        self.microbatches_per_epoch = math.ceil(
            int(train_samples) / int(cfg.training.batch_size)
        )
        self.optimizer_steps_per_epoch = math.ceil(
            self.microbatches_per_epoch / int(cfg.training.gradient_accumulation_steps)
        )
        super().__init__(cfg)

    def _maybe_compute_mel_on_gpu(self, batch):
        if self.cfg.data.input_type != "mel" or not bool(
            self.cfg.data.get("compute_mel_on_gpu", False)
        ):
            return batch

        speech = batch["speech"].to(self.device, dtype=torch.float32, non_blocking=True)
        speech = whisper.pad_or_trim(speech)
        mel = whisper.log_mel_spectrogram(
            speech,
            n_mels=int(self.cfg.data.mel_size),
        ).permute(0, 2, 1)

        batch = dict(batch)
        batch["speech"] = mel
        batch["speech_lengths"] = torch.full(
            (mel.shape[0],),
            mel.shape[1],
            dtype=torch.long,
            device=self.device,
        )
        return batch

    def forward(self, batch):
        return super().forward(self._maybe_compute_mel_on_gpu(batch))

    def _log_streaming_progress(self, batch_idx: int) -> None:
        completed_microbatches = int(self.global_step) * int(
            self.cfg.training.gradient_accumulation_steps
        ) + int(batch_idx) % int(self.cfg.training.gradient_accumulation_steps)
        true_epoch = completed_microbatches / max(1, self.microbatches_per_epoch)
        self.log(
            "streaming_true_epoch",
            true_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "streaming_microbatch_progress",
            float(completed_microbatches),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "streaming_optimizer_epoch",
            float(self.global_step) / max(1, self.optimizer_steps_per_epoch),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        self._log_streaming_progress(batch_idx)
        return super().training_step(batch, batch_idx)

    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
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
            f"Using HF streaming global cosine schedule: total_steps={self.total_optimizer_steps}, "
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


@hydra.main(version_base=None, config_path="../../configs", config_name="stage_2")
def main(cfg: DictConfig):
    pl.seed_everything(int(cfg.data.seed), workers=True)

    # Resolve (and validate) the data before loading the model, so a missing
    # local build fails fast.
    data_source = resolve_data_source(cfg.streaming)
    train_samples = data_source.train_samples
    schedule, val_interval = plan_schedule(cfg, data_source)
    total_optimizer_steps = schedule["total_optimizer_steps"]

    run_summary = {
        **data_source.summary(),
        "repo_id": str(cfg.streaming.repo_id) if data_source.mode == "hub" else None,
        "global_num_train_epochs": int(cfg.training.num_train_epochs),
        "optimizer_steps_per_epoch": schedule["optimizer_steps_per_epoch"],
        "global_total_optimizer_steps": total_optimizer_steps,
        "warmup_steps": schedule["warmup_steps"],
        "val_check_interval_batches": val_interval,
    }
    output_dir = to_absolute_path(str(cfg.logging.output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "streaming_run.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    module = HFStreamingTrainingModule(cfg, total_optimizer_steps, train_samples)
    data_module = HFStreamingSpeechDataModule(
        cfg, module.tokenizer, module.model.config, data_source=data_source
    )
    data_module.setup()
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
    final_dir = to_absolute_path(os.path.join(cfg.logging.output_dir, "final_model"))
    save_omni_speech_checkpoint(module, final_dir, metadata={"final": True, "streaming": True})
    if trainer.global_rank == 0:
        module.tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
