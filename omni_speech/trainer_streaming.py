"""Train OmniSpeech directly from Hugging Face parquet streaming.

This path is intentionally separate from trainer.py and trainer_shard.py. It
does not materialize FLAC files to disk; audio bytes are decoded from streamed
parquet rows in the dataloader.
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
from pathlib import Path
from typing import Iterable, Iterator

import hydra
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import torchaudio
import whisper
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import get_cosine_schedule_with_warmup

from omni_speech.datasets.preprocess import preprocess, preprocess_multimodal
from omni_speech.datasets.processing.materialize_shard_audio import (
    FIRST_TURN_PROMPT,
    list_matching_parquet_files,
)
from omni_speech.trainer import OmniSpeechTrainingModule, SpeechCollator
from omni_speech.utils import (
    build_callbacks,
    build_loggers,
    save_omni_speech_checkpoint,
)


def _partition_info() -> tuple[int, int]:
    """Return this process/worker partition id and the total partitions."""
    rank = 0
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    worker = get_worker_info()
    worker_id = worker.id if worker is not None else 0
    num_workers = worker.num_workers if worker is not None else 1

    return rank * num_workers + worker_id, world_size * num_workers


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


def _load_streaming_dataset(data_files: list[str], split_name: str, cache_dir: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HF streaming training requires the `datasets` package. "
            "Install it in the environment before running trainer_streaming.py."
        ) from exc

    return load_dataset(
        "parquet",
        data_files={split_name: data_files},
        split=split_name,
        streaming=True,
        cache_dir=cache_dir,
    )


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
        {"from": "human", "value": FIRST_TURN_PROMPT},
        {"from": "gpt", "value": assistant_text},
    ]


class HFStreamingSpeechDataset(IterableDataset):
    def __init__(
        self,
        data_files: list[str],
        tokenizer,
        model_config,
        split_name: str,
        cache_dir: str | None,
        seed: int,
        shuffle_buffer_size: int,
        max_samples: int | None,
        repeat: bool,
        input_type: str = "mel",
        mel_size: int = 128,
    ):
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.split_name = split_name
        self.cache_dir = cache_dir
        self.seed = int(seed)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.max_samples = max_samples
        self.repeat = repeat
        self.input_type = input_type
        self.mel_size = int(mel_size)
        self.data_args = type(
            "DataArgs",
            (),
            {"is_multimodal": True, "input_type": input_type, "mel_size": self.mel_size},
        )()

    def _row_to_item(self, row: dict) -> dict | None:
        audio = _decode_audio_value(row.get("audio"))
        if audio is None:
            return None

        conversations = _row_to_conversations(row)
        source = preprocess_multimodal([conversations], self.data_args)[0]
        text = preprocess([source], self.tokenizer, has_speech=True)

        if self.input_type == "raw":
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

    def _iter_one_pass(self, pass_idx: int) -> Iterator[dict]:
        dataset = _load_streaming_dataset(self.data_files, self.split_name, self.cache_dir)
        if self.shuffle_buffer_size > 0:
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer_size,
                seed=self.seed + pass_idx,
            )

        partition_id, num_partitions = _partition_info()
        emitted = 0
        for row_idx, row in enumerate(dataset):
            if row_idx % num_partitions != partition_id:
                continue
            item = self._row_to_item(row)
            if item is None:
                continue
            yield item
            emitted += 1
            if self.max_samples is not None and emitted >= self.max_samples:
                break

    def __iter__(self) -> Iterator[dict]:
        pass_idx = 0
        while True:
            yield from self._iter_one_pass(pass_idx)
            pass_idx += 1
            if not self.repeat:
                break


class HFStreamingSpeechDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, tokenizer, model_config):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.train_files: list[str] = []
        self.val_files: list[str] = []

    def setup(self, stage=None):
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

    def train_dataloader(self):
        self.setup()
        dataset = HFStreamingSpeechDataset(
            data_files=self.train_files,
            tokenizer=self.tokenizer,
            model_config=self.model_config,
            split_name="train",
            cache_dir=self.cfg.streaming.get("cache_dir"),
            seed=int(self.cfg.data.seed),
            shuffle_buffer_size=int(self.cfg.streaming.shuffle_buffer_size),
            max_samples=None,
            repeat=True,
            input_type=self.cfg.data.input_type,
            mel_size=int(self.cfg.data.mel_size),
        )
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.training.batch_size),
            num_workers=int(self.cfg.data.num_workers),
            collate_fn=SpeechCollator(self.tokenizer),
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        self.setup()
        if not self.val_files:
            return None
        dataset = HFStreamingSpeechDataset(
            data_files=self.val_files,
            tokenizer=self.tokenizer,
            model_config=self.model_config,
            split_name="validation",
            cache_dir=self.cfg.streaming.get("cache_dir"),
            seed=int(self.cfg.data.seed) + 10_000,
            shuffle_buffer_size=0,
            max_samples=int(self.cfg.streaming.validation_samples),
            repeat=False,
            input_type=self.cfg.data.input_type,
            mel_size=int(self.cfg.data.mel_size),
        )
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.training.batch_size),
            num_workers=int(self.cfg.data.num_workers),
            collate_fn=SpeechCollator(self.tokenizer),
            pin_memory=torch.cuda.is_available(),
        )


def compute_streaming_optimizer_steps(train_samples: int, batch_size: int, grad_accum: int, epochs: int) -> int:
    microbatches_per_epoch = math.ceil(train_samples / batch_size)
    return max(1, math.ceil(microbatches_per_epoch / grad_accum) * epochs)


def compute_validation_interval_batches(train_samples: int, batch_size: int, val_check_interval) -> int | float:
    if isinstance(val_check_interval, float) and 0.0 < val_check_interval < 1.0:
        microbatches_per_epoch = math.ceil(train_samples / batch_size)
        return max(1, int(round(microbatches_per_epoch * val_check_interval)))
    return val_check_interval


class HFStreamingTrainingModule(OmniSpeechTrainingModule):
    def __init__(self, cfg: DictConfig, total_optimizer_steps: int):
        self.total_optimizer_steps = total_optimizer_steps
        super().__init__(cfg)

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


@hydra.main(version_base=None, config_path="../configs", config_name="speech_streaming")
def main(cfg: DictConfig):
    pl.seed_everything(int(cfg.data.seed), workers=True)

    total_optimizer_steps = compute_streaming_optimizer_steps(
        int(cfg.streaming.train_samples),
        int(cfg.training.batch_size),
        int(cfg.training.gradient_accumulation_steps),
        int(cfg.training.num_train_epochs),
    )
    val_interval = compute_validation_interval_batches(
        int(cfg.streaming.train_samples),
        int(cfg.training.batch_size),
        cfg.training.val_check_interval,
    )

    run_summary = {
        "repo_id": str(cfg.streaming.repo_id),
        "global_num_train_epochs": int(cfg.training.num_train_epochs),
        "train_samples": int(cfg.streaming.train_samples),
        "validation_samples": int(cfg.streaming.validation_samples),
        "global_total_optimizer_steps": total_optimizer_steps,
        "val_check_interval_batches": val_interval,
    }
    output_dir = to_absolute_path(str(cfg.logging.output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "streaming_run.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    module = HFStreamingTrainingModule(cfg, total_optimizer_steps)
    data_module = HFStreamingSpeechDataModule(cfg, module.tokenizer, module.model.config)
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
