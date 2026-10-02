# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import csv
import os
import shutil
import logging
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from hydra.utils import to_absolute_path
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file


def load_audio_16k(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio as mono float32 and resample to the model's expected rate."""
    audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if file_sr != sample_rate:
        waveform = torch.from_numpy(audio).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, file_sr, sample_rate)
        audio = waveform.squeeze(0).numpy()
    return audio.astype(np.float32)


def optional_abs_path(path):
    if path in (None, ""):
        return None
    return to_absolute_path(str(path))


def model_dtype(precision):
    precision = str(precision)
    if "bf16" in precision:
        return torch.bfloat16
    if "16" in precision:
        return torch.float16
    return torch.float32


# --- Safetensors checkpoints (trainable LoRA + speech projector only) ---

_CHECKPOINT_MARKERS = (
    "adapter_model.safetensors",
    "speech_projector.safetensors",
    "trainable.safetensors",
)


def is_safetensors_checkpoint(path: str) -> bool:
    return os.path.isdir(path) and any(
        os.path.isfile(os.path.join(path, name)) for name in _CHECKPOINT_MARKERS
    )


def resolve_checkpoint_path(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path) or is_safetensors_checkpoint(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ranked = []
    legacy = []
    for name in os.listdir(path):
        sub = os.path.join(path, name)
        if os.path.isdir(sub) and is_safetensors_checkpoint(sub):
            score = float("inf")
            meta_path = os.path.join(sub, "checkpoint_meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                score = next((float(meta[k]) for k in ("val_loss", "train_loss_epoch") if k in meta), score)
            ranked.append((score, sub))
        elif name.endswith(".ckpt"):
            legacy.append(sub)

    if ranked:
        return sorted(ranked, key=lambda item: item[0])[0][1]
    if legacy:
        return sorted(legacy)[-1]
    raise FileNotFoundError(f"No checkpoints found under: {path}")


def resolve_training_state_path(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        if not path.endswith(".ckpt"):
            raise ValueError(f"Resume state must be a Lightning .ckpt file, got: {path}")
        return path
    if is_safetensors_checkpoint(path):
        raise ValueError(
            f"{path} is a weights-only safetensors checkpoint. "
            "Full resume requires a Lightning trainer-state .ckpt file."
        )
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Training state checkpoint not found: {path}")

    preferred = [
        os.path.join(path, "trainer_state", "last.ckpt"),
        os.path.join(path, "last.ckpt"),
    ]
    for candidate in preferred:
        if os.path.isfile(candidate):
            return candidate

    discovered = []
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith(".ckpt"):
                discovered.append(os.path.join(root, name))
    if discovered:
        return max(discovered, key=os.path.getmtime)
    raise FileNotFoundError(f"No Lightning trainer-state .ckpt file found under: {path}")

def _gather_param(param: torch.Tensor) -> torch.Tensor:
    return param.detach().cpu().clone()


def save_omni_speech_checkpoint(module, output_dir: str, metadata: Optional[Dict] = None) -> None:
    from peft import PeftModel

    os.makedirs(output_dir, exist_ok=True)
    model = module.model
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    if isinstance(model, PeftModel):
        if rank == 0:
            model.save_pretrained(output_dir, safe_serialization=True)
    else:
        trainable = {
            name: _gather_param(param)
            for name, param in module.named_parameters()
            if param.requires_grad
        }
        if trainable and rank == 0:
            save_file(trainable, os.path.join(output_dir, "trainable.safetensors"))

    inner = module._get_inner_speech_model()
    if getattr(inner, "speech_projector", None) is not None:
        state = {k: _gather_param(v) for k, v in inner.speech_projector.state_dict().items()}
        if rank == 0:
            save_file(state, os.path.join(output_dir, "speech_projector.safetensors"))

    if metadata is not None:
        if rank == 0:
            with open(os.path.join(output_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)


def load_omni_speech_checkpoint(module, checkpoint_dir: str, adapter_trainable: bool = False) -> None:
    from peft import PeftModel, load_peft_weights, set_peft_model_state_dict

    adapter = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    if os.path.isfile(adapter):
        if not isinstance(module.model, PeftModel):
            raise RuntimeError("LoRA checkpoint loaded into a non-LoRA model.")
        adapter_state = load_peft_weights(checkpoint_dir, device="cpu")
        incompat = set_peft_model_state_dict(module.model, adapter_state, adapter_name="default")
        if getattr(incompat, "unexpected_keys", None):
            raise RuntimeError(f"Unexpected LoRA keys when loading {checkpoint_dir}: {incompat.unexpected_keys}")
        if not adapter_trainable:
            for name, param in module.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False

    inner = module._get_inner_speech_model()
    projector = os.path.join(checkpoint_dir, "speech_projector.safetensors")
    if os.path.isfile(projector) and getattr(inner, "speech_projector", None) is not None:
        inner.speech_projector.load_state_dict(load_file(projector), strict=True)

    trainable = os.path.join(checkpoint_dir, "trainable.safetensors")
    if os.path.isfile(trainable) and not os.path.isfile(adapter):
        module.load_state_dict(load_file(trainable), strict=False)


class SafetensorsCheckpointCallback(Callback):
    def __init__(self, dirpath: str, monitor: str = "val_loss", mode: str = "min",
                 save_top_k: int = 1, save_last: bool = False):
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = int(save_top_k)
        self.save_last = bool(save_last)
        self.best_models: Dict[str, float] = {}
        os.makedirs(self.dirpath, exist_ok=True)

    def _metric(self, trainer) -> Optional[float]:
        if self.monitor not in trainer.callback_metrics:
            return None
        value = trainer.callback_metrics[self.monitor]
        return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)

    def _on_epoch_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "sanity_checking", False):
            return

        metric = self._metric(trainer)
        if metric is None:
            return

        tag = f"epoch={trainer.current_epoch}-step={trainer.global_step}-{self.monitor}={metric:.4f}"
        ckpt_dir = os.path.join(self.dirpath, tag)

        # Rank 0 writes the files; save_omni_speech_checkpoint coordinates the rest.
        save_omni_speech_checkpoint(
            pl_module, ckpt_dir,
            metadata={"epoch": int(trainer.current_epoch), "global_step": int(trainer.global_step), self.monitor: metric},
        )

        if trainer.global_rank != 0:
            return

        if not os.path.isdir(ckpt_dir):
            logging.warning("Skipping checkpoint bookkeeping because %s was not created.", ckpt_dir)
            return

        if self.save_last:
            last_dir = os.path.join(self.dirpath, "last")
            if os.path.isdir(last_dir):
                shutil.rmtree(last_dir)
            shutil.copytree(ckpt_dir, last_dir)

        self.best_models[ckpt_dir] = metric
        if self.save_top_k > 0:
            ranked = sorted(self.best_models.items(), key=lambda x: x[1], reverse=self.mode != "min")
            for stale_dir, _ in ranked[self.save_top_k:]:
                self.best_models.pop(stale_dir, None)
                if os.path.isdir(stale_dir):
                    shutil.rmtree(stale_dir)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._on_epoch_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "num_val_dataloaders", 0) > 0:
            return
        self._on_epoch_end(trainer, pl_module)


class ResumeStateCheckpointCallback(Callback):
    """Persist one rolling Lightning checkpoint with optimizer/scheduler state."""

    def __init__(self, dirpath: str, filename: str = "last.ckpt"):
        self.dirpath = dirpath
        self.filename = filename
        os.makedirs(self.dirpath, exist_ok=True)

    @property
    def path(self) -> str:
        return os.path.join(self.dirpath, self.filename)

    def _save(self, trainer) -> None:
        trainer.save_checkpoint(self.path, weights_only=False)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._save(trainer)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "num_val_dataloaders", 0) > 0:
            return
        self._save(trainer)

    def on_exception(self, trainer, pl_module, exception) -> None:
        self._save(trainer)

    def on_train_end(self, trainer, pl_module) -> None:
        self._save(trainer)


def build_loggers(cfg: DictConfig):
    output_dir = to_absolute_path(str(cfg.logging.output_dir))
    loggers = []

    if cfg.logging.get("tensorboard", True):
        loggers.append(TensorBoardLogger(save_dir=output_dir, name="tensorboard"))
    if cfg.logging.get("wandb", False):
        from pytorch_lightning.loggers import WandbLogger

        loggers.append(
            WandbLogger(
                project=cfg.logging.wandb_project,
                id=cfg.logging.get("wandb_run_id"),
                name=cfg.logging.get("wandb_run_name"),
                save_dir=output_dir,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        )

    return loggers


def build_callbacks(cfg: DictConfig, has_validation: bool):
    output_dir = to_absolute_path(str(cfg.logging.output_dir))
    monitor = cfg.logging.checkpoint_monitor
    if not has_validation and str(monitor).startswith("val_"):
        monitor = "train_loss_epoch"

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    checkpoint_format = str(cfg.logging.get("checkpoint_format", "safetensors")).lower()
    if checkpoint_format == "lightning":
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch={epoch}-step={step}-loss={%s:.4f}" % monitor,
            monitor=monitor,
            mode="min",
            save_top_k=cfg.logging.save_top_k,
            save_last=cfg.logging.save_last,
            save_weights_only=bool(cfg.logging.get("save_weights_only", False)),
        )
    else:
        checkpoint_callback = SafetensorsCheckpointCallback(
            dirpath=checkpoint_dir,
            monitor=monitor,
            mode="min",
            save_top_k=cfg.logging.save_top_k,
            save_last=cfg.logging.save_last,
        )

    log_path = os.path.join(
        output_dir,
        str(cfg.logging.get("log_file", "logs/train.log")),
    )
    csv_path = None
    if cfg.logging.get("csv", True):
        csv_path = os.path.join(
            output_dir,
            str(cfg.logging.get("csv_metrics_file", "csv/metrics.csv")),
        )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
        LocalMetricsLogCallback(
            log_path=log_path,
            csv_path=csv_path,
            every_n_steps=int(cfg.logging.get("local_log_every_n_steps", 500)),
        ),
    ]
    if bool(cfg.logging.get("save_resume_state", False)):
        resume_dir = os.path.join(
            output_dir,
            str(cfg.logging.get("resume_state_dir", "trainer_state")),
        )
        callbacks.append(ResumeStateCheckpointCallback(dirpath=resume_dir))
    return callbacks


# --- Local metrics table (logs/train.log + csv/metrics.csv) ---

def format_metrics_table(headers, rows):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def _row(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [sep, _row(headers), sep]
    lines.extend(_row(row) for row in rows)
    lines.append(sep)
    return "\n".join(lines)


class LocalMetricsLogCallback(Callback):
    """Write a local metrics table to ``train.log`` and ``metrics.csv`` (rank 0 only).

    Each row summarises a *window* of optimizer steps: the steps completed since the
    previous row (or since the start of the fit). ``train_loss`` is the unweighted mean
    of ``train_loss_accum`` over the optimizer steps in the window, where
    ``train_loss_accum`` is the module's per-optimizer-step loss (mean of the
    gradient-accumulation micro-batch losses of that step, averaged over ranks).
    ``train_steps_in_window`` is the number of optimizer steps in that mean.

    Rows are written:
      * after every real validation (``val_loss`` filled; sanity checks are skipped),
      * every ``every_n_steps`` optimizer steps if no validation ran at that step
        (training-only row, empty ``val_loss``),
      * at the end of a completed training epoch if steps remain in the window.
    ``step`` is ``trainer.global_step``; ``lr`` is the learning rate of the first
    param group used by the last optimizer step of the window.
    """

    TRAIN_LOSS_KEY = "train_loss_accum"
    VAL_LOSS_KEY = "val_loss"
    HEADERS = ("epoch", "step", "train_loss", "train_steps_in_window", "val_loss", "lr")

    def __init__(self, log_path: str, csv_path: str | None = None, every_n_steps: int = 500):
        self.log_path = log_path
        self.csv_path = csv_path
        self.every_n_steps = int(every_n_steps)
        self.rows = []
        self._window = []
        self._last_step = 0
        self._last_lr = None
        self._row_due = False

    @staticmethod
    def _metric(trainer, key: str):
        if key not in trainer.callback_metrics:
            return None
        value = trainer.callback_metrics[key]
        return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)

    @staticmethod
    def _optimizer_lr(trainer):
        optimizers = trainer.optimizers
        if not optimizers:
            return None
        return float(optimizers[0].param_groups[0]["lr"])

    @staticmethod
    def _fmt(value, digits=4):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    @staticmethod
    def _csv_cell(value):
        return "" if value is None else value

    def _write_csv_header(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.HEADERS)

    def _append_csv_row(self, row) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([self._csv_cell(row[h]) for h in self.HEADERS])
            f.flush()
            os.fsync(f.fileno())

    def _load_existing_csv_rows(self) -> bool:
        """On resume, keep the rows of the interrupted run if the CSV has our header."""
        if not self.csv_path or not os.path.isfile(self.csv_path):
            return False
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != self.HEADERS:
                return False
            self.rows = [
                {key: (None if value == "" else value) for key, value in row.items()}
                for row in reader
            ]
        return True

    def _write_log_table(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        table_rows = [
            [
                row["epoch"],
                row["step"],
                self._fmt(row["train_loss"]),
                row["train_steps_in_window"],
                self._fmt(row["val_loss"]),
                self._fmt(row["lr"], digits=9),
            ]
            for row in self.rows
        ]
        tmp_path = self.log_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("Training log\n")
            f.write(
                f"train_loss = mean of per-optimizer-step {self.TRAIN_LOSS_KEY} over the "
                "train_steps_in_window optimizer steps since the previous row\n"
            )
            f.write(format_metrics_table(self.HEADERS, table_rows))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.log_path)

    def _emit_row(self, trainer, val_loss=None) -> None:
        window = self._window
        row = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "train_loss": sum(window) / len(window) if window else None,
            "train_steps_in_window": len(window),
            "val_loss": val_loss,
            "lr": self._last_lr if self._last_lr is not None else self._optimizer_lr(trainer),
        }
        self._window = []
        self._row_due = False
        self.rows.append(row)
        if self.csv_path:
            self._append_csv_row(row)
        self._write_log_table()

    def on_fit_start(self, trainer, pl_module) -> None:
        self.rows = []
        self._window = []
        self._last_lr = None
        self._row_due = False
        if trainer.global_rank != 0:
            return
        resumed = bool(trainer.ckpt_path) and self._load_existing_csv_rows()
        if self.csv_path and not resumed:
            self._write_csv_header()
        self._write_log_table()

    def on_train_start(self, trainer, pl_module) -> None:
        # Runs after a resume checkpoint has restored global_step.
        self._last_step = trainer.global_step

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        # A periodic row is written one batch late so that a validation running at the
        # same step can claim the window instead (its row then carries the train loss).
        if trainer.global_rank == 0 and self._row_due and self._window:
            self._emit_row(trainer)
        self._row_due = False

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        # Lightning steps "interval: step" schedulers before on_train_batch_end, so the
        # LR the step actually uses must be read here.
        self._last_lr = float(optimizer.param_groups[0]["lr"])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step == self._last_step:
            return  # gradient-accumulation micro-batch, no optimizer step
        self._last_step = trainer.global_step
        if trainer.global_rank != 0:
            return
        train_loss = self._metric(trainer, self.TRAIN_LOSS_KEY)
        if train_loss is not None:
            self._window.append(train_loss)
        if self.every_n_steps > 0 and trainer.global_step % self.every_n_steps == 0:
            self._row_due = True

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0 or trainer.sanity_checking or trainer.state.fn != "fit":
            return
        self._emit_row(trainer, val_loss=self._metric(trainer, self.VAL_LOSS_KEY))

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.global_rank == 0 and self._window:
            self._emit_row(trainer)

    def on_train_end(self, trainer, pl_module) -> None:
        if trainer.global_rank == 0 and self._window:
            self._emit_row(trainer)
