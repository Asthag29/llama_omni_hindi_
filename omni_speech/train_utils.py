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
import math
import os
import shutil
import logging
import time
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from hydra.utils import to_absolute_path
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor
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


def missing_checkpoint_message(path: str) -> str:
    """Error text for a missing checkpoint, naming a killed run's best-so-far weights."""
    message = f"Checkpoint not found: {path}"
    stable_best = stable_best_checkpoint_path(os.path.dirname(os.path.abspath(path)))
    if is_safetensors_checkpoint(stable_best):
        message += (
            f". The run in {os.path.dirname(os.path.abspath(path))} did not finish, but its "
            f"best-so-far weights exist: pass {stable_best} instead "
            f"(e.g. model.init_checkpoint={stable_best})."
        )
    return message


def resolve_checkpoint_path(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path) or is_safetensors_checkpoint(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(missing_checkpoint_message(path))

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
                score = next((float(meta[k]) for k in ("val_loss", "train_loss_epoch")
                              if meta.get(k) is not None), score)
            if not math.isfinite(score):
                score = float("inf")  # NaN would break the ordering
            ranked.append((score, sub))
        elif name.endswith(".ckpt"):
            legacy.append(sub)

    if ranked:
        return sorted(ranked, key=lambda item: item[0])[0][1]
    if legacy:
        return sorted(legacy)[-1]
    raise FileNotFoundError(f"No checkpoints found under: {path}")


def _gather_param(param: torch.Tensor) -> torch.Tensor:
    return param.detach().cpu().clone()


def save_omni_speech_checkpoint(module, output_dir: str, metadata: Optional[Dict] = None) -> None:
    from peft import PeftModel

    model = module.model
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

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


# --- Best / last weights during a fit, best_model / final_model after it ---
#
# Layout of <output_dir> while a fit runs (rank 0 writes; only trainable weights):
#   checkpoints/step=<N>/          real weight directories, at most two at a time
#   checkpoints/best -> step=<N>   best-so-far by logging.checkpoint_monitor (lower is
#                                  better), from the first real validation onwards;
#                                  a relative symlink swapped atomically (os.replace)
#   checkpoints/last -> step=<M>   weights of the latest validation (logging.save_last);
#                                  points at the same directory as `best` when the
#                                  latest validation was the best one
# After a completed fit (finalize_fit_outputs):
#   best_model/    best-by-validation weights (moved out of checkpoints/, not copied)
#   final_model/   last weights (moved from checkpoints/ when the last validation ran at
#                  the final step, hard-linked when that was also the best, else written)
# and checkpoints/ is removed.

CHECKPOINTS_DIRNAME = "checkpoints"
BEST_LINK_NAME = "best"
LAST_LINK_NAME = "last"
BEST_MODEL_DIRNAME = "best_model"
FINAL_MODEL_DIRNAME = "final_model"
CHECKPOINT_META_FILENAME = "checkpoint_meta.json"
_INCOMPLETE_PREFIX = ".incomplete-"


def stable_best_checkpoint_path(output_dir: str) -> str:
    """The best-so-far weights of a (possibly still running or killed) fit."""
    return os.path.join(output_dir, CHECKPOINTS_DIRNAME, BEST_LINK_NAME)


def save_trainable_weights(module, output_dir: str, metadata: Optional[Dict] = None) -> None:
    """Write the module's trainable weights (and checkpoint_meta.json) into ``output_dir``.

    A module may define ``save_trainable_weights(output_dir, metadata)`` to override
    how its weights are written; otherwise ``save_omni_speech_checkpoint`` is used.
    """
    hook = getattr(module, "save_trainable_weights", None)
    if callable(hook):
        hook(output_dir, metadata)
    else:
        save_omni_speech_checkpoint(module, output_dir, metadata=metadata)


def _write_json_atomic(path: str, data: Dict) -> None:
    # Write a new inode and rename it over the old one, so a hard-linked sibling
    # directory never sees the change.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _point_symlink(link_path: str, target_name: str) -> None:
    """Atomically make ``link_path`` a relative symlink to ``target_name``."""
    tmp_path = link_path + ".tmp"
    if os.path.lexists(tmp_path):
        os.remove(tmp_path)
    os.symlink(target_name, tmp_path)
    os.replace(tmp_path, link_path)


def _remove_path(path: str) -> None:
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def _link_or_copy_tree(src: str, dst: str, skip=()) -> None:
    """Recreate ``src`` at ``dst`` with hard links (no extra disk); copy if linking fails."""
    os.makedirs(dst)
    for name in os.listdir(src):
        if name in skip:
            continue
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)
        if os.path.isdir(src_path):
            _link_or_copy_tree(src_path, dst_path, skip)
            continue
        try:
            os.link(src_path, dst_path)
        except OSError:
            logging.warning("Hard link %s -> %s failed; copying instead.", src_path, dst_path)
            shutil.copy2(src_path, dst_path)


class BestWeightsCheckpointCallback(Callback):
    """Keep the best-so-far and the latest weights of a fit on disk (see layout above).

    Only real validations count (sanity checks are ignored). Lower ``monitor`` is
    better. At most two weight directories exist at any time, including while a new
    one is being written: the previous ``last`` is deleted before the new weights are
    written unless it is also the best.
    """

    def __init__(self, output_dir: str, monitor: str = "val_loss", save_last: bool = True):
        self.output_dir = output_dir
        self.dirpath = os.path.join(output_dir, CHECKPOINTS_DIRNAME)
        self.monitor = monitor
        self.save_last = bool(save_last)
        self.best: Optional[Dict] = None  # {"name", "epoch", "global_step", monitor}
        self.last: Optional[Dict] = None

    @property
    def best_path(self) -> str:
        return os.path.join(self.dirpath, BEST_LINK_NAME)

    @property
    def last_path(self) -> str:
        return os.path.join(self.dirpath, LAST_LINK_NAME)

    def _metric(self, trainer) -> Optional[float]:
        if self.monitor not in trainer.callback_metrics:
            return None
        value = trainer.callback_metrics[self.monitor]
        return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)

    def on_fit_start(self, trainer, pl_module) -> None:
        self.best = None
        self.last = None
        if trainer.global_rank != 0:
            return
        if os.path.isdir(self.dirpath) and os.listdir(self.dirpath):
            aside = f"{self.dirpath}_previous_run_{time.strftime('%Y%m%d-%H%M%S')}"
            os.rename(self.dirpath, aside)
            print(
                f"WARNING: {self.dirpath} was not empty (an earlier run's weights); moved it "
                f"to {aside}. Delete it when no longer needed: it uses disk quota.",
                flush=True,
            )
        os.makedirs(self.dirpath, exist_ok=True)

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or trainer.state.fn != "fit":
            return
        metric = self._metric(trainer)
        if metric is None:
            logging.warning("%s was not logged during validation; no checkpoint written.", self.monitor)
            return
        improved = self._is_improvement(metric)
        if not math.isfinite(metric):
            if improved:
                logging.warning(
                    "%s=%s at step %d is not finite; keeping these weights only as a placeholder "
                    "best until the first finite %s.",
                    self.monitor, metric, int(trainer.global_step), self.monitor,
                )
            else:
                logging.warning(
                    "%s=%s at step %d is not finite; it never counts as an improvement%s.",
                    self.monitor, metric, int(trainer.global_step),
                    " (saved as last only)" if self.save_last else "",
                )
        if improved or self.save_last:
            self._save(trainer, pl_module, metric, improved)

    def _is_improvement(self, metric: float) -> bool:
        """Lower is better. A non-finite metric only fills an empty best (as a
        placeholder); any finite metric beats a non-finite best."""
        if self.best is None:
            return True
        if not math.isfinite(metric):
            return False
        best = self.best[self.monitor]
        return best is None or not math.isfinite(best) or metric < best

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Without validation there is no "best"; keep the latest weights only.
        if trainer.enable_validation or not self.save_last:
            return
        self._save(trainer, pl_module, None, improved=False)

    def _new_name(self, step: int) -> str:
        taken = {record["name"] for record in (self.best, self.last) if record is not None}
        name, suffix = f"step={step}", 1
        while name in taken:
            suffix += 1
            name = f"step={step}-{suffix}"
        return name

    def _save(self, trainer, pl_module, metric: Optional[float], improved: bool) -> None:
        step, epoch = int(trainer.global_step), int(trainer.current_epoch)
        name = self._new_name(step)
        record = {"name": name, "epoch": epoch, "global_step": step, self.monitor: metric}
        rank0 = trainer.global_rank == 0
        stale_last = self.last is not None and (self.best is None or self.last["name"] != self.best["name"])
        tmp_dir = os.path.join(self.dirpath, _INCOMPLETE_PREFIX + name)

        if rank0:
            # Free the previous `last` before writing, so at most two copies ever exist.
            if stale_last:
                _remove_path(self.last_path)
                _remove_path(os.path.join(self.dirpath, self.last["name"]))
            _remove_path(tmp_dir)

        metadata = {"epoch": epoch, "global_step": step}
        if metric is not None:
            metadata[self.monitor] = metric
        save_trainable_weights(pl_module, tmp_dir, metadata=metadata)

        if rank0:
            if not os.path.isdir(tmp_dir):
                raise RuntimeError(f"Saving trainable weights did not create {tmp_dir}.")
            os.rename(tmp_dir, os.path.join(self.dirpath, name))
            if improved:
                _point_symlink(self.best_path, name)
            if self.save_last:
                _point_symlink(self.last_path, name)
            if improved and self.best is not None:
                _remove_path(os.path.join(self.dirpath, self.best["name"]))  # superseded best
            if improved:
                note = "" if math.isfinite(metric) else " (non-finite placeholder until a finite value)"
                print(
                    f"New best weights: {self.monitor}={metric:.4f} at step {step} -> {self.best_path}{note}",
                    flush=True,
                )

        if improved:
            self.best = record
        self.last = record if self.save_last else None


def find_best_weights_callback(trainer) -> Optional[BestWeightsCheckpointCallback]:
    for callback in trainer.callbacks:
        if isinstance(callback, BestWeightsCheckpointCallback):
            return callback
    return None


def _fmt_metric(value) -> str:
    return "not validated at this step" if value is None else f"{value:.4f}"


def finalize_fit_outputs(trainer, module, output_dir: str, final_metadata: Optional[Dict] = None,
                         tokenizer=None) -> Dict:
    """After a completed ``trainer.fit``: write ``best_model/`` and ``final_model/``.

    ``best_model`` is moved out of ``checkpoints/`` (no copy). ``final_model`` is the
    latest ``checkpoints/`` entry if it was taken at the final step (moved, or
    hard-linked when it is also the best), otherwise freshly written. If no validation
    ran, ``best_model`` is a hard-linked twin of ``final_model``. Previous
    ``best_model``/``final_model`` directories are replaced and ``checkpoints/`` is
    removed. Prints one summary line and returns the summary.
    """
    output_dir = os.path.abspath(output_dir)
    callback = find_best_weights_callback(trainer)
    monitor = callback.monitor if callback is not None else "val_loss"
    best = callback.best if callback is not None else None
    last = callback.last if callback is not None else None
    ckpt_dir = callback.dirpath if callback is not None else os.path.join(output_dir, CHECKPOINTS_DIRNAME)
    best_dir = os.path.join(output_dir, BEST_MODEL_DIRNAME)
    final_dir = os.path.join(output_dir, FINAL_MODEL_DIRNAME)
    rank0 = trainer.global_rank == 0

    step = int(trainer.global_step)
    # A checkpoints/ entry taken at the final step already holds the final weights.
    reuse = next((r for r in (last, best) if r is not None and r["global_step"] == step), None)
    final_record = dict(reuse) if reuse is not None else {
        "epoch": int(trainer.current_epoch), "global_step": step, monitor: None,
    }
    final_record.pop("name", None)

    if rank0:
        _remove_path(best_dir)
        _remove_path(final_dir)
        if last is not None and last is not reuse and (best is None or last["name"] != best["name"]):
            _remove_path(os.path.join(ckpt_dir, last["name"]))  # stale; frees disk first
        if best is not None:
            os.rename(os.path.join(ckpt_dir, best["name"]), best_dir)  # move, not copy
        if reuse is not None:
            if best is not None and reuse["name"] == best["name"]:
                _link_or_copy_tree(best_dir, final_dir, skip=(CHECKPOINT_META_FILENAME,))
            else:
                os.rename(os.path.join(ckpt_dir, reuse["name"]), final_dir)

    final_meta = {**final_record, "final": True, **(final_metadata or {})}
    if reuse is None:
        save_trainable_weights(module, final_dir, metadata=final_meta)

    if rank0:
        _write_json_atomic(os.path.join(final_dir, CHECKPOINT_META_FILENAME), final_meta)
        if best is None:
            _link_or_copy_tree(final_dir, best_dir, skip=(CHECKPOINT_META_FILENAME,))
            best_meta = {**final_record, "best": True, "no_validation": True}
        else:
            best_meta = {k: v for k, v in best.items() if k != "name"}
            best_meta["best"] = True
        _write_json_atomic(os.path.join(best_dir, CHECKPOINT_META_FILENAME), best_meta)

        for link in (BEST_LINK_NAME, LAST_LINK_NAME):
            if os.path.islink(os.path.join(ckpt_dir, link)):
                os.remove(os.path.join(ckpt_dir, link))
        if os.path.isdir(ckpt_dir) and not os.listdir(ckpt_dir):
            os.rmdir(ckpt_dir)

        if tokenizer is not None:
            tokenizer.save_pretrained(best_dir)
            tokenizer.save_pretrained(final_dir)

    if best is None:
        summary_line = (
            f"No validation ran: best_model = final weights (step {step}). "
            f"best_model: {best_dir}; final_model: {final_dir}"
        )
    else:
        summary_line = (
            f"Best weights: step {best['global_step']} (epoch {best['epoch']}), "
            f"{monitor}={_fmt_metric(best[monitor])} -> {best_dir}; "
            f"final weights: step {step}, {monitor}={_fmt_metric(final_record.get(monitor))} -> {final_dir}"
        )
        if best[monitor] is not None and not math.isfinite(best[monitor]):
            summary_line += f" (WARNING: every validation {monitor} was non-finite)"

    if rank0:
        print(summary_line, flush=True)
    return {"best": best_meta if rank0 else None, "final": final_meta, "line": summary_line}


def has_any_logger(cfg: DictConfig) -> bool:
    """Whether ``build_loggers`` creates at least one logger."""
    return bool(cfg.logging.get("tensorboard", True)) or bool(cfg.logging.get("wandb", False))


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


# Local metrics table, relative to logging.output_dir.
LOCAL_LOG_FILE = os.path.join("logs", "train.log")
LOCAL_CSV_FILE = os.path.join("csv", "metrics.csv")
# The checkpoint metric (lower is better). checkpoint_meta.json stores it under this
# key, and resolve_checkpoint_path / inference rank checkpoints by it.
CHECKPOINT_MONITOR = "val_loss"


def build_callbacks(cfg: DictConfig, has_validation: bool):
    """Checkpoint (best/last weights by val_loss), LR monitor (if a logger is on) and local metrics-table callbacks.

    Reads ``logging.save_last`` and ``logging.csv``.
    """
    output_dir = to_absolute_path(str(cfg.logging.output_dir))
    if not has_validation:
        logging.warning(
            "No validation data: no best-by-validation weights are tracked; "
            "best_model will equal the final weights."
        )
    checkpoint_callback = BestWeightsCheckpointCallback(
        output_dir=output_dir,
        monitor=CHECKPOINT_MONITOR,
        save_last=bool(cfg.logging.get("save_last", True)),
    )

    log_path = os.path.join(output_dir, LOCAL_LOG_FILE)
    csv_path = os.path.join(output_dir, LOCAL_CSV_FILE) if cfg.logging.get("csv", True) else None
    callbacks = [checkpoint_callback]
    # Lightning refuses a LearningRateMonitor without a logger; with W&B and TensorBoard
    # both off the local train.log / metrics.csv still record the LR.
    if has_any_logger(cfg):
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(
        LocalMetricsLogCallback(
            log_path=log_path,
            csv_path=csv_path,
            every_n_steps=int(cfg.logging.get("local_log_every_n_steps", 500)),
        )
    )
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
        if self.csv_path:
            self._write_csv_header()
        self._write_log_table()

    def on_train_start(self, trainer, pl_module) -> None:
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
