import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, random_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from omni_speech.constants import IGNORE_INDEX
from omni_speech.datasets.preprocess import preprocess
from omni_speech.model.language_model.omni_speech_llama import (
    OmniSpeechConfig,
    OmniSpeechLlamaForCausalLM,
)
from omni_speech.utils import (
    build_callbacks,
    build_loggers,
    model_dtype,
    optional_abs_path,
    save_omni_speech_checkpoint,
)


def load_json_array_maybe_prefixed(path: str) -> List[Dict]:
    text = Path(path).read_text(encoding="utf-8")
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not locate a JSON array in {path}")
    return json.loads(text[start : end + 1])


class TextConversationDataset(Dataset):
    """Text-only dataset that normalizes either messages or conversations format."""

    def __init__(self, data_path: str, tokenizer, model_config):
        self.data_path = optional_abs_path(data_path)
        self.tokenizer = tokenizer
        self.model_config = model_config

        raw_samples = load_json_array_maybe_prefixed(self.data_path)
        self.samples = self._normalize_samples(raw_samples)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _normalize_turns(turns: List[Dict]) -> Optional[List[Dict]]:
        normalized = []
        for turn in turns:
            if "from" in turn and "value" in turn:
                role = turn["from"]
                value = str(turn["value"]).strip()
            elif "role" in turn and "content" in turn:
                role = {"user": "human", "assistant": "gpt"}.get(turn["role"])
                value = str(turn["content"]).strip()
            else:
                return None

            if role not in {"human", "gpt"} or not value:
                return None

            normalized.append({"from": role, "value": value})

        if not normalized or normalized[0]["from"] != "human":
            return None

        assistant_idx = next(
            (idx for idx, turn in enumerate(normalized[1:], start=1) if turn["from"] == "gpt"),
            None,
        )
        if assistant_idx is None:
            return None

        return [normalized[0], normalized[assistant_idx]]

    def _normalize_samples(self, samples: List[Dict]) -> List[Dict]:
        usable = []
        skipped_bad_format = 0

        for idx, item in enumerate(samples):
            turns = None
            if "conversations" in item:
                turns = self._normalize_turns(item["conversations"])
            elif "messages" in item:
                turns = self._normalize_turns(item["messages"])

            if turns is None:
                skipped_bad_format += 1
                continue

            usable.append(
                {
                    "id": item.get("id", f"sample-{idx}"),
                    "conversations": turns,
                }
            )

        print(
            f"Loaded {len(usable)} text-only samples from {self.data_path} "
            f"(skipped {skipped_bad_format} malformed samples)."
        )
        return usable

    def __getitem__(self, index):
        item = self.samples[index]
        source = copy.deepcopy(item["conversations"])
        text = preprocess([source], self.tokenizer, has_speech=False)
        return {
            "input_ids": text["input_ids"].squeeze(0),
            "labels": text["labels"].squeeze(0),
        }


class TextCollator:
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

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class TextLengthBucketSampler(Sampler):
    """Bucket samples of similar text lengths to reduce padding waste."""

    def __init__(self, dataset, batch_size: int, bucket_size_multiplier: int = 10):
        self.batch_size = batch_size
        self.dataset = dataset

        lengths = []
        for i in range(len(dataset)):
            item = self._sample_metadata(dataset, i)
            text_len = sum(len(turn.get("value", "")) for turn in item.get("conversations", []))
            lengths.append((i, text_len))

        lengths.sort(key=lambda x: x[1])

        bucket_size = batch_size * bucket_size_multiplier
        self.buckets = []
        for start in range(0, len(lengths), bucket_size):
            bucket = [idx for idx, _ in lengths[start : start + bucket_size]]
            self.buckets.append(bucket)

    def __iter__(self):
        import random

        bucket_order = list(range(len(self.buckets)))
        random.shuffle(bucket_order)
        for bi in bucket_order:
            bucket = self.buckets[bi][:]
            random.shuffle(bucket)
            yield from bucket

    def __len__(self):
        return sum(len(bucket) for bucket in self.buckets)

    @staticmethod
    def _sample_metadata(dataset, index: int) -> Dict:
        while isinstance(dataset, Subset):
            index = int(dataset.indices[index])
            dataset = dataset.dataset
        return dataset.samples[index]


class TextDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, tokenizer, model_config):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.train_dataset = None
        self.val_dataset = None

    def _fraction_subset(self, dataset, fraction: float, seed_offset: int, name: str):
        if fraction >= 1.0:
            return dataset
        if fraction <= 0.0:
            raise ValueError(f"data.{name}_fraction must be in (0, 1], got {fraction}")

        subset_size = max(1, int(round(len(dataset) * fraction)))
        generator = torch.Generator().manual_seed(int(self.cfg.data.seed) + seed_offset)
        indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
        print(f"Using {subset_size}/{len(dataset)} {name} samples ({fraction:.0%}).")
        return Subset(dataset, indices)

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        dataset = TextConversationDataset(
            self.cfg.data.json_path,
            self.tokenizer,
            self.model_config,
        )

        val_split = float(self.cfg.data.validation_split)
        if len(dataset) < 2 or val_split <= 0:
            self.train_dataset = self._fraction_subset(
                dataset,
                float(self.cfg.data.get("train_fraction", 1.0)),
                101,
                "train",
            )
            self.val_dataset = None
            return

        val_size = int(round(len(dataset) * val_split))
        val_size = max(1, min(len(dataset) - 1, val_size))
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(int(self.cfg.data.seed))
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )
        self.train_dataset = self._fraction_subset(
            train_dataset,
            float(self.cfg.data.get("train_fraction", 1.0)),
            101,
            "train",
        )
        self.val_dataset = self._fraction_subset(
            val_dataset,
            float(self.cfg.data.get("val_fraction", 1.0)),
            202,
            "val",
        )

    def train_dataloader(self):
        batch_size = int(self.cfg.training.batch_size)
        if batch_size > 1:
            sampler = TextLengthBucketSampler(self.train_dataset, batch_size)
            return DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=self.cfg.data.num_workers,
                collate_fn=TextCollator(self.tokenizer),
                pin_memory=torch.cuda.is_available(),
            )
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            collate_fn=TextCollator(self.tokenizer),
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
            collate_fn=TextCollator(self.tokenizer),
            pin_memory=torch.cuda.is_available(),
        )


class BackboneTrainingModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.tokenizer, self.model = self._load_model_and_tokenizer()
        self._maybe_apply_lora()
        self._configure_trainable_parameters()
        self._maybe_enable_gradient_checkpointing()
        self._promote_trainable_params_to_fp32()

    def _get_inner_speech_model(self):
        model = self.model
        if hasattr(model, "get_model"):
            return model.get_model()
        base = getattr(model, "base_model", None)
        if base is not None and hasattr(base, "model"):
            inner = base.model
            if hasattr(inner, "get_model"):
                return inner.get_model()
            return inner
        raise AttributeError("Could not locate inner OmniSpeech model.")

    def _maybe_apply_lora(self):
        tune_llm = bool(self.cfg.training.get("tune_llm_backbone", False))
        use_lora = bool(self.cfg.training.get("use_lora", False))
        if not (tune_llm and use_lora):
            return

        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=int(self.cfg.training.get("lora_r", 64)),
            lora_alpha=int(self.cfg.training.get("lora_alpha", 64)),
            lora_dropout=float(self.cfg.training.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _configure_trainable_parameters(self):
        tune_projector = bool(self.cfg.training.get("tune_speech_projector", False))
        tune_llm = bool(self.cfg.training.get("tune_llm_backbone", False))
        tune_encoder = bool(self.cfg.training.get("tune_speech_encoder", False))
        use_lora = bool(self.cfg.training.get("use_lora", False)) and tune_llm

        if not use_lora:
            for param in self.model.parameters():
                param.requires_grad = False

        inner_model = self._get_inner_speech_model()
        if tune_llm and not use_lora:
            for name, param in inner_model.named_parameters():
                if name.startswith("speech_encoder") or name.startswith("speech_projector"):
                    continue
                param.requires_grad = True
            for param in self.model.lm_head.parameters():
                param.requires_grad = True

        if getattr(inner_model, "speech_projector", None) is not None:
            for param in inner_model.speech_projector.parameters():
                param.requires_grad = tune_projector

        speech_encoder = inner_model.get_speech_encoder()
        if speech_encoder is not None:
            if tune_encoder:
                speech_encoder.train()
            else:
                speech_encoder.eval()
            for param in speech_encoder.parameters():
                param.requires_grad = tune_encoder

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            "Trainable parameters: "
            f"{trainable:,} / {total:,} ({100 * trainable / total:.2f}%) "
            f"[projector={tune_projector}, llm={tune_llm}, lora={use_lora}, encoder={tune_encoder}]"
        )

    def _load_model_and_tokenizer(self):
        config_path = to_absolute_path(str(self.cfg.model.config_path))
        model_base = optional_abs_path(self.cfg.model.get("model_base"))
        tokenizer_path = optional_abs_path(self.cfg.model.get("tokenizer_path"))

        config = OmniSpeechConfig.from_pretrained(config_path)
        config.tokenizer_model_max_length = self.cfg.model.model_max_length
        config.tokenizer_padding_side = "right"
        config._attn_implementation = str(
            self.cfg.training.get("attn_implementation", "sdpa")
        )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = self.cfg.model.model_max_length

        model = OmniSpeechLlamaForCausalLM.from_pretrained(
            model_base or config_path,
            config=config,
            torch_dtype=model_dtype(self.cfg.training.precision),
            low_cpu_mem_usage=False,
        )

        return tokenizer, model

    def _maybe_enable_gradient_checkpointing(self):
        enabled = bool(self.cfg.training.get("gradient_checkpointing", False))
        if not enabled:
            return
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        print("Gradient checkpointing enabled.")

    def _promote_trainable_params_to_fp32(self):
        tune_llm = bool(self.cfg.training.get("tune_llm_backbone", False))
        precision = str(self.cfg.training.precision)
        if tune_llm or "bf16" in precision:
            return
        for param in self.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            speech=None,
            speech_lengths=None,
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
            sync_dist=True,
            batch_size=batch["input_ids"].shape[0],
        )
        return loss

    def on_before_optimizer_step(self, optimizer):
        grad_norms = [
            param.grad.detach().float().norm(2)
            for param in self.parameters()
            if param.requires_grad and param.grad is not None
        ]
        if not grad_norms:
            return

        global_grad_norm = torch.stack(grad_norms).norm(2)
        self.log(
            "grad_norm_global",
            global_grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
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

@hydra.main(version_base=None, config_path="../configs", config_name="backbone")
def main(cfg: DictConfig):
    pl.seed_everything(int(cfg.data.seed), workers=True)

    module = BackboneTrainingModule(cfg)
    data_module = TextDataModule(cfg, module.tokenizer, module.model.config)
    data_module.setup()
    has_validation = data_module.val_dataset is not None
    callbacks = build_callbacks(cfg, has_validation)
    val_check_interval = cfg.training.val_check_interval

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
        callbacks=callbacks,
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=val_check_interval,
        fast_dev_run=cfg.training.fast_dev_run,
        enable_checkpointing=False,
    )

    trainer.fit(module, datamodule=data_module)
    final_dir = to_absolute_path(os.path.join(cfg.logging.output_dir, "final_model"))
    save_omni_speech_checkpoint(module, final_dir, metadata={"final": True})
    if trainer.global_rank == 0:
        module.tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
