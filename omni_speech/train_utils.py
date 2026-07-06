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
import sys
import logging
import logging.handlers
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
import transformers
from hydra.utils import to_absolute_path
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file

from omni_speech.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


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


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def _copy_param(param: torch.Tensor) -> torch.Tensor:
    return param.detach().cpu().clone()


def get_peft_state(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: _copy_param(v) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: _copy_param(v) for k, v in to_return.items()}
    return to_return


def get_speech_projector_state(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: _copy_param(v) for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    speech_keywords = ['speech_projector', 'speech_encoder']
    for name, module in model.named_modules():
        if any(speech_keyword in name for speech_keyword in speech_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_speech_projector", False):
        # Only save projector
        keys_to_match = ['speech_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_speech_projector_state(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                speech_projector_folder = os.path.join(parent_folder, "speech_projector")
                os.makedirs(speech_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(speech_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'speech_projector.bin'))
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


def lengths_to_mask(lens):
    return ~lengths_to_padding_mask(lens)


#* disabling the redundant torch default initialization to accelerate model creation.
def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]




def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


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
        metric = self._metric(trainer)
        if metric is None:
            return

        tag = f"epoch={trainer.current_epoch}-step={trainer.global_step}-{self.monitor}={metric:.4f}"
        ckpt_dir = os.path.join(self.dirpath, tag)

        # All ranks participate in save (ZeRO-3 needs gather on all ranks),
        # but only rank 0 writes files — handled inside save_omni_speech_checkpoint.
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
        LocalMetricsLogCallback(log_path=log_path, csv_path=csv_path),
    ]
    if bool(cfg.logging.get("save_resume_state", False)):
        resume_dir = os.path.join(
            output_dir,
            str(cfg.logging.get("resume_state_dir", "trainer_state")),
        )
        callbacks.append(ResumeStateCheckpointCallback(dirpath=resume_dir))
    return callbacks


# --- Training log (loguru) ---

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


def setup_training_log(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(f"training_metrics:{os.path.abspath(log_path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)

    class _LoggerAdapter:
        def info(self, message, *args):
            if args:
                message = message.format(*args)
            logger.info(message)

    return _LoggerAdapter()


class LocalMetricsLogCallback(Callback):
    """Text logs plus epoch CSV metrics."""

    LOG_HEADERS = ("epoch", "step", "train_loss_epoch", "val_loss", "lr")
    SUMMARY_HEADERS = ("epoch", "train_loss_epoch", "val_loss")
    CSV_HEADERS = ("epoch", "lr", "train_loss", "val_loss")

    def __init__(self, log_path: str, csv_path: str | None = None):
        self.log_path = log_path
        self.csv_path = csv_path
        self.epoch_records = {}
        self._logger = None

    def _write_epoch_csv(self) -> None:
        if not self.csv_path:
            return
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADERS)
            for _, record in sorted(self.epoch_records.items()):
                writer.writerow([
                    record["epoch"],
                    "" if record["lr"] is None else record["lr"],
                    "" if record["train_loss"] is None else record["train_loss"],
                    "" if record["val_loss"] is None else record["val_loss"],
                ])

    def _write_epoch_log(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        rows = []
        for _, record in sorted(self.epoch_records.items()):
            rows.append([
                record["epoch"],
                record["step"],
                self._fmt(record["train_loss"]),
                self._fmt(record["val_loss"]),
                self._fmt(record["lr"], digits=9),
            ])
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("Training log\n")
            f.write(format_metrics_table(self.LOG_HEADERS, rows))
            f.write("\n")

    def _logger_instance(self):
        if self._logger is None:
            self._logger = setup_training_log(self.log_path)
        return self._logger

    @staticmethod
    def _metric(trainer, key: str):
        if key not in trainer.callback_metrics:
            return None
        value = trainer.callback_metrics[key]
        return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)

    @staticmethod
    def _current_lr(trainer):
        optimizers = getattr(trainer, "optimizers", None)
        if not optimizers:
            return None
        optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers
        return float(optimizer.param_groups[0]["lr"])

    @staticmethod
    def _fmt(value, digits=4):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    def _record_epoch(self, trainer) -> None:
        if trainer.global_rank != 0:
            return

        train_loss = self._metric(trainer, "train_loss_epoch")
        val_loss = self._metric(trainer, "val_loss")
        lr = self._metric(trainer, "lr-AdamW") or self._current_lr(trainer)
        if train_loss is None and val_loss is None:
            return

        epoch = trainer.current_epoch
        key = (epoch, trainer.global_step)
        existing = self.epoch_records.get(key, {})
        self.epoch_records[key] = {
            "epoch": epoch,
            "step": trainer.global_step,
            "lr": lr if lr is not None else existing.get("lr"),
            "train_loss": train_loss if train_loss is not None else existing.get("train_loss"),
            "val_loss": val_loss if val_loss is not None else existing.get("val_loss"),
        }
        self._write_epoch_log()
        self._write_epoch_csv()

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0 or not self.csv_path:
            return
        self.epoch_records = {}
        self._write_epoch_log()
        self._write_epoch_csv()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_rank != 0:
            return
        train_loss = self._metric(trainer, "train_loss")
        if train_loss is None:
            return
        # Per-step metrics still go to Lightning/W&B/CSV summaries; we no longer
        # mirror them into a separate local train_perstep.log file.

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._record_epoch(trainer)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "num_val_dataloaders", 0) > 0:
            return
        self._record_epoch(trainer)

    def on_fit_end(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0 or not self.epoch_records:
            return
        summary_rows = []
        for _, record in sorted(self.epoch_records.items()):
            summary_rows.append([
                record["epoch"],
                self._fmt(record["train_loss"]),
                self._fmt(record["val_loss"]),
            ])
        self._logger_instance().info(
            "Metrics (train / val per epoch)\n{}",
            format_metrics_table(self.SUMMARY_HEADERS, summary_rows),
        )