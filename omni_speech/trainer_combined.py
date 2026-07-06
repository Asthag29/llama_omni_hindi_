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
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, random_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from omni_speech.constants import IGNORE_INDEX
from omni_speech.datasets.preprocess import preprocess, preprocess_multimodal
from omni_speech.train_utils import (
    build_callbacks,
    build_loggers,
    load_omni_speech_checkpoint,
    load_audio_16k,
    model_dtype,
    optional_abs_path,
    resolve_checkpoint_path,
    resolve_training_state_path,
    save_omni_speech_checkpoint,
)
from omni_speech.model.language_model.omni_speech_llama import (
    OmniSpeechConfig,
    OmniSpeechLlamaForCausalLM,
)

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
        self.data_path = optional_abs_path(data_path)
        self.data_dir = os.path.dirname(self.data_path)
        self.audio_root = optional_abs_path(audio_root)
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.input_type = input_type
        self.mel_size = mel_size

        with open(self.data_path, "r") as f:
            samples = json.load(f)

        # Read the flac directory once — just filenames, not file contents.
        # This turns 2 filesystem calls per sample into 1 fast hash lookup.
        root = self.audio_root or self.data_dir
        self._existing_files = set(os.listdir(root)) if os.path.isdir(root) else set()

        self.samples = self._filter_usable_samples(samples)

        # Simple namespace matching what preprocess_multimodal expects,
        # fed from the same Hydra config values passed to this class.
        self.data_args = type("DataArgs", (), {
            "is_multimodal": True,
            "input_type": input_type,
            "mel_size": mel_size,
        })()

    def __len__(self):
        return len(self.samples)

    def _has_usable_audio(self, item):
        speech_path = item.get("speech")
        if not speech_path:
            return False
        filename = os.path.basename(speech_path)
        return filename in self._existing_files

    def _filter_usable_samples(self, samples): #* if conversation is not in the item, skip it
        usable = []
        skipped_missing_audio = 0
        skipped_no_conversation = 0

        for item in samples:
            if "conversations" not in item:
                skipped_no_conversation += 1
                continue
            if not self._has_usable_audio(item):
                skipped_missing_audio += 1
                continue
            usable.append(item)

        print(
            "Loaded "
            f"{len(usable)} usable speech samples from {self.data_path} "
            f"(skipped {skipped_missing_audio} missing/empty audio, "
            f"{skipped_no_conversation} without conversations)."
        )
        return usable

    def _resolve_speech_path(self, speech_path):
        speech_path = os.path.expanduser(speech_path)
        if os.path.isabs(speech_path):
            return speech_path
        root = self.audio_root or self.data_dir
        return os.path.join(root, speech_path)

    #* volume normalization
    def _load_speech(self, speech_path):
        speech = load_audio_16k(self._resolve_speech_path(speech_path))
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
            [instance["input_ids"] for instance in instances], #instance is one sample from the dataset
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


class LengthBucketSampler(Sampler):
    """Groups samples of similar text length into batches to minimize padding waste.

    Works with both raw Dataset and Subset (from random_split).
    Shuffles *between* buckets each epoch so training order varies,
    but samples *within* a batch have similar lengths.
    """

    def __init__(self, dataset, batch_size: int, bucket_size_multiplier: int = 10):
        self.batch_size = batch_size
        self.dataset = dataset

        lengths = []
        for i in range(len(dataset)):
            idx = dataset.indices[i] if hasattr(dataset, "indices") else i
            base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
            item = base_dataset.samples[idx]
            text_len = sum(len(turn.get("value", "")) for turn in item.get("conversations", []))
            lengths.append((i, text_len))

        lengths.sort(key=lambda x: x[1])

        # Group sorted indices into buckets, then shuffle buckets each epoch.
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
        return sum(len(b) for b in self.buckets)


class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, tokenizer, model_config):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _fraction_subset(self, dataset, fraction: float, seed_offset: int, name: str):
        if dataset is None or fraction >= 1.0:
            return dataset
        if fraction <= 0.0:
            raise ValueError(f"data.{name}_fraction must be in (0, 1], got {fraction}")

        subset_size = max(1, int(round(len(dataset) * fraction)))
        generator = torch.Generator().manual_seed(int(self.cfg.data.seed) + seed_offset)
        indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
        print(f"Using {subset_size}/{len(dataset)} {name} samples ({fraction:.0%}).")
        return Subset(dataset, indices)

    @staticmethod
    def _bounded_split_size(dataset_size: int, fraction: float, min_size: int, max_size: int) -> int:
        if dataset_size <= 0 or fraction <= 0.0 or max_size <= 0:
            return 0
        size = int(round(dataset_size * fraction))
        return max(min_size, min(max_size, size))

    def setup(self, stage=None):   #!need to understand this
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
        test_split = float(self.cfg.data.get("test_split", 0.0))
        if val_split < 0 or test_split < 0:
            raise ValueError("data.validation_split and data.test_split must be non-negative.")
        if val_split + test_split >= 1.0:
            raise ValueError("data.validation_split + data.test_split must be < 1.0.")

        if len(dataset) < 2 or (val_split <= 0 and test_split <= 0):
            self.train_dataset = self._fraction_subset(
                dataset,
                float(self.cfg.data.get("train_fraction", 1.0)),
                101,
                "train",
            )
            self.val_dataset = None
            self.test_dataset = None
            return

        dataset_size = len(dataset)
        test_size = self._bounded_split_size(
            dataset_size,
            test_split,
            min_size=1 if test_split > 0 else 0,
            max_size=max(0, dataset_size - 1),
        )
        remaining_after_test = dataset_size - test_size
        val_size = self._bounded_split_size(
            dataset_size,
            val_split,
            min_size=1 if val_split > 0 else 0,
            max_size=max(0, remaining_after_test - 1),
        )
        train_size = dataset_size - val_size - test_size

        generator = torch.Generator().manual_seed(int(self.cfg.data.seed))  #manual_seed is used to ensure that the random split is reproducible
        splits = [train_size]
        if val_size > 0:
            splits.append(val_size)
        if test_size > 0:
            splits.append(test_size)
        subsets = random_split(dataset, splits, generator=generator)

        self.train_dataset = self._fraction_subset(
            subsets[0],
            float(self.cfg.data.get("train_fraction", 1.0)),
            101,
            "train",
        )
        next_idx = 1
        if val_size > 0:
            self.val_dataset = self._fraction_subset(
                subsets[next_idx],
                float(self.cfg.data.get("val_fraction", 1.0)),
                202,
                "val",
            )
            next_idx += 1
        else:
            self.val_dataset = None
        if test_size > 0:
            self.test_dataset = self._fraction_subset(
                subsets[next_idx],
                float(self.cfg.data.get("test_fraction", 1.0)),
                303,
                "test",
            )
        else:
            self.test_dataset = None

    def train_dataloader(self):
        batch_size = self.cfg.training.batch_size
        if batch_size > 1:
            sampler = LengthBucketSampler(self.train_dataset, batch_size)
            return DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=self.cfg.data.num_workers,
                collate_fn=SpeechCollator(self.tokenizer),
                pin_memory=torch.cuda.is_available(),
            )
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
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

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
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
        self.speech_dtype = model_dtype(cfg.training.precision)
        self._maybe_apply_lora()
        self._maybe_load_init_checkpoint()
        self._configure_trainable_parameters()
        self._maybe_enable_gradient_checkpointing()
        self._promote_trainable_params_to_fp32()
        self._accumulated_microbatch_losses = []

    def _get_inner_speech_model(self): #! need to check this for training with print statements
        model = self.model
        if hasattr(model, "get_model"):
            return model.get_model()
        base = getattr(model, "base_model", None)
        if base is not None and hasattr(base, "model"): #! after lora wrapper self.model is a peftobject which has a base_model attribute
            inner = base.model  #! base.model is the actual model without the lora wrapper
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

            r=int(self.cfg.training.get("lora_r", 64)),  #! need to change this
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

    def _maybe_load_init_checkpoint(self):
        init_path = self.cfg.model.get("init_checkpoint")
        if not init_path:
            return
        checkpoint_path = resolve_checkpoint_path(optional_abs_path(init_path))
        print(f"Initializing speech trainer weights from {checkpoint_path}")
        load_omni_speech_checkpoint(self, checkpoint_path, adapter_trainable=True)

    def _configure_trainable_parameters(self):
        tune_projector = bool(self.cfg.training.get("tune_speech_projector", True))
        tune_llm = bool(self.cfg.training.get("tune_llm_backbone", False))
        tune_encoder = bool(self.cfg.training.get("tune_speech_encoder", False))
        use_lora = bool(self.cfg.training.get("use_lora", False)) and tune_llm

        if not use_lora: #no lora
            for param in self.model.parameters():
                param.requires_grad = False  #frozen

        inner_model = self._get_inner_speech_model()
        if tune_llm and not use_lora:
            for name, param in inner_model.named_parameters():
                if name.startswith("speech_encoder") or name.startswith("speech_projector"):
                    continue
                param.requires_grad = True
            for param in self.model.lm_head.parameters():
                param.requires_grad = True

        if tune_projector and getattr(inner_model, "speech_projector", None) is not None:
            for param in inner_model.speech_projector.parameters():
                param.requires_grad = True

        speech_encoder = inner_model.get_speech_encoder()
        if speech_encoder is not None:
            if tune_encoder:
                speech_encoder.train()
                for param in speech_encoder.parameters():
                    param.requires_grad = True
            else:
                speech_encoder.eval()
                for param in speech_encoder.parameters():
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            "Trainable parameters: "
            f"{trainable:,} / {total:,} ({100 * trainable / total:.2f}%) "
            f"[projector={tune_projector}, llm={tune_llm}, lora={use_lora}, encoder={tune_encoder}]"
        )

    def _load_model_and_tokenizer(self):
        config_path = to_absolute_path(str(self.cfg.model.config_path))
        model_base = optional_abs_path(self.cfg.model.get("model_base"))  #llama model path
        tokenizer_path = optional_abs_path(self.cfg.model.get("tokenizer_path"))
        # tokenizer_path = tokenizer_path or model_base or config_path

        config = OmniSpeechConfig.from_pretrained(config_path)
        config.tokenizer_model_max_length = self.cfg.model.model_max_length
        config.tokenizer_padding_side = "right"
        config._attn_implementation = str(
            self.cfg.training.get("attn_implementation", "sdpa")
        )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False) # tokenization type (fast = rust implementation/ slow = python implementation)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = self.cfg.model.model_max_length

        model = OmniSpeechLlamaForCausalLM.from_pretrained(
            model_base or config_path,
            config=config,
            torch_dtype=model_dtype(self.cfg.training.precision),
            low_cpu_mem_usage=False,  #todo : can experiment with True for better performance
        )

        return tokenizer, model

    def _maybe_enable_gradient_checkpointing(self): #recomputes missing activations in the forward pass when needed
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
        # Only needed for small trainable subsets under fp16 AMP.
        # Full LLM fine-tuning must stay in bf16/fp16 or it OOMs immediately.
        tune_llm = bool(self.cfg.training.get("tune_llm_backbone", False))
        precision = str(self.cfg.training.precision)
        if tune_llm or "bf16" in precision:
            return
        for param in self.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    def _keep_speech_encoder_eval(self):
        if bool(self.cfg.training.get("tune_speech_encoder", False)):
            return
        speech_encoder = self._get_inner_speech_model().get_speech_encoder()
        if speech_encoder is None:
            return
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
        self._accumulated_microbatch_losses.append(loss.detach())
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

    def _log_param_group_stats(self, name: str, grad_norms: List[torch.Tensor], weight_norms: List[torch.Tensor]) -> None:
        if not grad_norms:
            return
        grad_norm = torch.stack(grad_norms).norm(2)
        weight_norm = torch.stack(weight_norms).norm(2)
        grad_weight_ratio = grad_norm / weight_norm.clamp_min(1e-12)
        self.log(
            f"grad_norm_{name}",
            grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            f"grad_weight_ratio_{name}",
            grad_weight_ratio,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

    def on_before_optimizer_step(self, optimizer):
        if self._accumulated_microbatch_losses:
            effective_batch_loss = torch.stack(self._accumulated_microbatch_losses).mean()
            self.log(
                "train_loss_accum",
                effective_batch_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
            self._accumulated_microbatch_losses.clear()

        trainable_params = [
            param
            for param in self.parameters()
            if param.requires_grad and param.grad is not None
        ]
        if not trainable_params:
            return

        grad_norms = [param.grad.detach().float().norm(2) for param in trainable_params]
        weight_norms = [param.detach().float().norm(2) for param in trainable_params]
        self._log_param_group_stats("global", grad_norms, weight_norms)

        lora_a_grad_norms = []
        lora_a_weight_norms = []
        lora_b_grad_norms = []
        lora_b_weight_norms = []
        projector_grad_norms = []
        projector_weight_norms = []
        for name, param in self.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            grad_norm = param.grad.detach().float().norm(2)
            weight_norm = param.detach().float().norm(2)
            if "lora_A" in name:
                lora_a_grad_norms.append(grad_norm)
                lora_a_weight_norms.append(weight_norm)
            elif "lora_B" in name:
                lora_b_grad_norms.append(grad_norm)
                lora_b_weight_norms.append(weight_norm)
            elif "speech_projector" in name:
                projector_grad_norms.append(grad_norm)
                projector_weight_norms.append(weight_norm)

        self._log_param_group_stats("lora_A", lora_a_grad_norms, lora_a_weight_norms)
        self._log_param_group_stats("lora_B", lora_b_grad_norms, lora_b_weight_norms)
        self._log_param_group_stats("speech_projector", projector_grad_norms, projector_weight_norms)

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

@hydra.main(version_base=None, config_path="../configs", config_name="combined")
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
        enable_checkpointing=False,
    )

    resume_cfg = cfg.get("resume")
    resume_path = resume_cfg.get("path") if resume_cfg is not None else None
    ckpt_path = None
    if resume_path:
        ckpt_path = resolve_training_state_path(resume_path)
        print(f"Resuming trainer state from {ckpt_path}")

    trainer.fit(module, datamodule=data_module, ckpt_path=ckpt_path)
    final_dir = to_absolute_path(os.path.join(cfg.logging.output_dir, "final_model"))
    save_omni_speech_checkpoint(module, final_dir, metadata={"final": True})
    if trainer.global_rank == 0:
        module.tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
