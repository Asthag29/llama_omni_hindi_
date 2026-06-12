import copy
import json
import os
from typing import Dict, List

import hydra
import pytorch_lightning as pl
import torch
import whisper
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from omni_speech.arguments import DataArguments
from omni_speech.constants import IGNORE_INDEX
from omni_speech.datasets.preprocess import preprocess, preprocess_multimodal
from omni_speech.model.language_model.omni_speech_llama import (
    OmniSpeechConfig,
    OmniSpeechLlamaForCausalLM,
)


def _optional_abs_path(path):
    if path in (None, ""):
        return None
    return to_absolute_path(str(path))


def _model_dtype(precision):
    precision = str(precision)
    if "bf16" in precision:
        return torch.bfloat16
    if "16" in precision:
        return torch.float16
    return torch.float32


class SpeechDataset(Dataset):
    """Dataset for speech conversation samples; batching is handled by the collator."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_config,
        input_type="mel",
        mel_size=128,
        audio_root=None,
    ):
        self.data_path = _optional_abs_path(data_path)
        self.data_dir = os.path.dirname(self.data_path)
        self.audio_root = _optional_abs_path(audio_root)
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.input_type = input_type
        self.mel_size = mel_size

        with open(self.data_path, "r") as f:
            self.samples = json.load(f)

        self.data_args = DataArguments(
            is_multimodal=True,
            input_type=input_type,
            mel_size=mel_size,
        )

    def __len__(self):
        return len(self.samples)

    def _resolve_speech_path(self, speech_path):
        speech_path = os.path.expanduser(speech_path)
        if os.path.isabs(speech_path):
            return speech_path
        root = self.audio_root or self.data_dir
        return os.path.join(root, speech_path)

    def _load_speech(self, speech_path):
        speech = whisper.load_audio(self._resolve_speech_path(speech_path))
        if self.input_type == "raw":
            speech = torch.from_numpy(speech)
            if getattr(self.model_config, "speech_normalize", False):
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
            return speech, speech.shape[0]

        if self.input_type != "mel":
            raise ValueError(f"Unsupported input_type: {self.input_type}")

        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)
        return speech, speech.shape[0]

    def __getitem__(self, index):
        item = self.samples[index]
        source = copy.deepcopy(item["conversations"])
        source = preprocess_multimodal([source], self.data_args)[0]
        text = preprocess([source], self.tokenizer, has_speech=True)
        speech, speech_length = self._load_speech(item["speech"])

        return {
            "input_ids": text["input_ids"].squeeze(0),
            "labels": text["labels"].squeeze(0),
            "speech": speech,
            "speech_length": torch.tensor(speech_length, dtype=torch.long),
        }


class SpeechCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, instances: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [instance["input_ids"] for instance in instances],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            [instance["labels"] for instance in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        attention_mask = input_ids.ne(self.pad_token_id)
        speech = pad_sequence(
            [instance["speech"] for instance in instances],
            batch_first=True,
            padding_value=0,
        )
        speech_lengths = torch.stack([instance["speech_length"] for instance in instances])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "speech": speech,
            "speech_lengths": speech_lengths,
        }


class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, tokenizer, model_config):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        dataset = SpeechDataset(
            self.cfg.data.json_path,
            self.tokenizer,
            self.model_config,
            input_type=self.cfg.data.input_type,
            mel_size=self.cfg.data.mel_size,
            audio_root=self.cfg.data.get("audio_root"),
        )

        val_split = float(self.cfg.data.validation_split)
        if len(dataset) < 2 or val_split <= 0:
            self.train_dataset = dataset
            self.val_dataset = None
            return

        val_size = int(round(len(dataset) * val_split))
        val_size = max(1, min(len(dataset) - 1, val_size))
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(int(self.cfg.data.seed))
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            collate_fn=SpeechCollator(self.tokenizer),
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            collate_fn=SpeechCollator(self.tokenizer),
            pin_memory=torch.cuda.is_available(),
        )


class OmniSpeechTrainingModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.tokenizer, self.model = self._load_model_and_tokenizer()
        self.speech_dtype = _model_dtype(cfg.training.precision)
        self._freeze_speech_encoder()
        self._unfreeze_speech_projector()

    def _load_model_and_tokenizer(self):
        config_path = to_absolute_path(str(self.cfg.model.config_path))
        model_base = _optional_abs_path(self.cfg.model.get("model_base"))
        tokenizer_path = _optional_abs_path(self.cfg.model.get("tokenizer_path"))
        tokenizer_path = tokenizer_path or model_base or config_path

        config = OmniSpeechConfig.from_pretrained(config_path)
        config.tokenizer_model_max_length = self.cfg.model.model_max_length
        config.tokenizer_padding_side = "right"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = self.cfg.model.model_max_length

        model = OmniSpeechLlamaForCausalLM.from_pretrained(
            model_base or config_path,
            config=config,
            torch_dtype=_model_dtype(self.cfg.training.precision),
            low_cpu_mem_usage=False,
        )

        return tokenizer, model

    def _freeze_speech_encoder(self):
        speech_encoder = self.model.get_model().get_speech_encoder()
        if speech_encoder is None:
            return
        speech_encoder.eval()
        for param in speech_encoder.parameters():
            param.requires_grad = False

    def _unfreeze_speech_projector(self):
        for param in self.model.get_model().speech_projector.parameters():
            param.requires_grad = True

    def _keep_speech_encoder_eval(self):
        speech_encoder = self.model.get_model().get_speech_encoder()
        if speech_encoder is not None:
            speech_encoder.eval()

    def on_train_epoch_start(self):
        self._keep_speech_encoder_eval()

    def on_validation_epoch_start(self):
        self._keep_speech_encoder_eval()

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            speech=batch["speech"].to(dtype=self.speech_dtype),
            speech_lengths=batch["speech_lengths"],
            use_cache=False,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["input_ids"].shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["input_ids"].shape[0],
        )
        return loss

    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

        if self.cfg.training.lr_scheduler_type != "cosine":
            return optimizer

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = int(total_steps * float(self.cfg.training.warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def build_loggers(cfg: DictConfig):
    output_dir = to_absolute_path(str(cfg.logging.output_dir))
    loggers = []

    if cfg.logging.csv:
        loggers.append(CSVLogger(save_dir=output_dir, name="csv"))
    if cfg.logging.tensorboard:
        loggers.append(TensorBoardLogger(save_dir=output_dir, name="tensorboard"))
    if cfg.logging.wandb:
        from pytorch_lightning.loggers import WandbLogger

        loggers.append(
            WandbLogger(
                project=cfg.logging.wandb_project,
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

    return [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="epoch={epoch}-step={step}-loss={%s:.4f}" % monitor,
            monitor=monitor,
            mode="min",
            save_top_k=cfg.logging.save_top_k,
            save_last=cfg.logging.save_last,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


@hydra.main(config_path="..", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(int(cfg.data.seed), workers=True)

    module = OmniSpeechTrainingModule(cfg)
    data_module = SpeechDataModule(cfg, module.tokenizer, module.model.config)
    data_module.setup()
    has_validation = data_module.val_dataset is not None

    trainer = pl.Trainer(
        default_root_dir=to_absolute_path(str(cfg.logging.output_dir)),
        max_epochs=cfg.training.num_train_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.strategy,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        gradient_clip_val=cfg.training.max_grad_norm,
        logger=build_loggers(cfg),
        callbacks=build_callbacks(cfg, has_validation),
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        fast_dev_run=cfg.training.fast_dev_run,
    )

    trainer.fit(module, datamodule=data_module)
    module.model.save_pretrained(to_absolute_path(os.path.join(cfg.logging.output_dir, "final_model")))
    module.tokenizer.save_pretrained(to_absolute_path(os.path.join(cfg.logging.output_dir, "final_model")))


if __name__ == "__main__":
    main()
