import copy
import os
from typing import Dict, List, Optional

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from omni_speech.constants import IGNORE_INDEX
from omni_speech.datasets.json_utils import load_json_array_maybe_prefixed
from omni_speech.datasets.preprocess import preprocess
from omni_speech.datasets.splits import (
    format_split_table,
    load_id_list,
    partition_by_ids,
)
from omni_speech.model.language_model.omni_speech_llama import (
    OmniSpeechConfig,
    OmniSpeechLlamaForCausalLM,
)
from omni_speech.train_utils import (
    build_callbacks,
    build_loggers,
    finalize_fit_outputs,
    model_dtype,
    optional_abs_path,
)


class TextConversationDataset(Dataset):
    """Text-only dataset that normalizes either messages or conversations format."""

    def __init__(self, data_path: str, tokenizer):
        self.data_path = optional_abs_path(data_path)
        self.tokenizer = tokenizer

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
    def __init__(self, cfg: DictConfig, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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

    def _split_id_path(self, key: str) -> str:
        path = optional_abs_path(self.cfg.data.get(key))
        if path is None:
            raise ValueError(
                f"data.{key} is not set in the config. Stage 1 uses the fixed id lists in "
                "data/splits/; there is no random-split fallback."
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"data.{key} points to a missing file: {path}")
        return path

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        validation_ids = load_id_list(self._split_id_path("validation_ids_path"))
        test_ids = load_id_list(self._split_id_path("test_ids_path"))

        dataset = TextConversationDataset(
            self.cfg.data.json_path,
            self.tokenizer,
        )

        indexed = [
            {"id": sample["id"], "index": index}
            for index, sample in enumerate(dataset.samples)
        ]
        parts = partition_by_ids(indexed, validation_ids, test_ids)
        print("Stage-1 split (fixed id lists from data.validation_ids_path / data.test_ids_path):")
        print(
            format_split_table(
                {name: [item["id"] for item in items] for name, items in parts.items()}
            )
        )
        train_dataset, val_dataset, test_dataset = (
            Subset(dataset, [item["index"] for item in parts[name]])
            for name in ("train", "validation", "test")
        )

        self.train_dataset = self._fraction_subset(
            train_dataset,
            float(self.cfg.data.get("train_fraction", 1.0)),
            101,
            "train",
        )
        self.val_dataset = (
            self._fraction_subset(
                val_dataset,
                float(self.cfg.data.get("val_fraction", 1.0)),
                202,
                "val",
            )
            if len(val_dataset) > 0
            else None
        )
        self.test_dataset = test_dataset if len(test_dataset) > 0 else None

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

    def _eval_dataloader(self, dataset):
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            collate_fn=TextCollator(self.tokenizer),
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return self._eval_dataloader(self.val_dataset)

    def test_dataloader(self):
        """Held-out test split; only used by ``trainer.test``, never during ``fit``."""
        return self._eval_dataloader(self.test_dataset)


def promote_trainable_params_to_fp32(module, training_cfg) -> None:
    """Keep the trainable subset (LoRA adapter / speech projector) in fp32.

    The optimizer is built from these parameters, so AdamW's moments are fp32 too;
    bf16 master weights round small updates away and freeze Adam's second moment.
    The frozen base keeps the dtype it was loaded in. Call after LoRA is attached
    and requires_grad is configured, and before any checkpoint is loaded into the
    trainable parameters or the optimizer is built.
    """
    tune_llm = bool(training_cfg.get("tune_llm_backbone", False))
    use_lora = bool(training_cfg.get("use_lora", False)) and tune_llm
    # Full LLM fine-tuning (no LoRA) must stay in bf16/fp16 or it OOMs immediately:
    # fp32 weights, grads and Adam moments for ~8B parameters do not fit.
    full_finetune = tune_llm and not use_lora
    trainable = [p for p in module.parameters() if p.requires_grad]
    if not full_finetune:
        for param in trainable:
            if param.is_floating_point():
                param.data = param.data.float()
    trainable_dtypes = sorted({str(p.dtype) for p in trainable})
    frozen_dtypes = sorted({str(p.dtype) for p in module.parameters() if not p.requires_grad})
    print(
        f"Trainable params: {sum(p.numel() for p in trainable):,} in {trainable_dtypes}; "
        f"frozen base in {frozen_dtypes}"
    )


def check_trainable_params_fp32(module, training_cfg) -> None:
    """Fail fast if the trainable LoRA adapter / speech projector is not fp32.

    ``training.precision: "bf16-true"`` / ``"16-true"`` make Lightning cast the whole
    module to half precision when fit starts, undoing
    ``promote_trainable_params_to_fp32``. Call from ``on_fit_start`` (after Lightning
    has converted the module). Full fine-tuning without LoRA is exempt, as in
    ``promote_trainable_params_to_fp32``.
    """
    tune_llm = bool(training_cfg.get("tune_llm_backbone", False))
    use_lora = bool(training_cfg.get("use_lora", False)) and tune_llm
    if tune_llm and not use_lora:
        return
    wrong = [
        (name, param.dtype)
        for name, param in module.named_parameters()
        if param.requires_grad and param.is_floating_point() and param.dtype != torch.float32
    ]
    if wrong:
        dtypes = sorted({str(dtype) for _, dtype in wrong})
        raise RuntimeError(
            f"training.precision={str(training_cfg.get('precision'))!r} cast {len(wrong)} trainable "
            f"parameters (LoRA adapter / speech projector) to {dtypes}, e.g. {wrong[0][0]}. "
            "They and the optimizer state must stay fp32. Use training.precision: "
            "\"bf16-mixed\" (the base model and forward pass still run in bf16); "
            "do not use \"bf16-true\" or \"16-true\"."
        )


def set_training_modes(module) -> None:
    """Put the LLM (and its LoRA wrappers) in train mode; keep a frozen speech encoder in eval.

    ``from_pretrained`` returns the model in eval mode, and Lightning never calls
    ``.train()`` on submodules: it records each submodule's mode before validation
    and restores exactly that afterwards. ``LlamaModel`` only applies gradient
    checkpointing when ``self.training`` is True, so without this the configured
    checkpointing is silently skipped. Call at the end of ``__init__`` (the mode
    Lightning then records and restores) and again when training starts.
    The LLM's only dropouts are the LoRA ``lora_dropout`` modules (already in train
    mode, being new modules) and ``attention_dropout`` (0.0 in the Llama config).
    """
    module.model.train()
    if bool(module.cfg.training.get("tune_speech_encoder", False)):
        return
    speech_encoder = module._get_inner_speech_model().get_speech_encoder()
    if speech_encoder is not None:
        speech_encoder.eval()


def report_training_modes(module) -> None:
    """Print (once per fit) and check the modes that matter at the first training step."""
    inner = module._get_inner_speech_model()
    speech_encoder = inner.get_speech_encoder()
    llm_training = bool(module.model.training and inner.training)
    checkpointing = bool(getattr(inner, "gradient_checkpointing", False) and inner.training)
    encoder_training = None if speech_encoder is None else bool(speech_encoder.training)
    print(
        f"LLM training mode: {llm_training}; gradient checkpointing active: {checkpointing}; "
        f"speech encoder training mode: {encoder_training}",
        flush=True,
    )
    if not llm_training:
        raise RuntimeError("The language model is in eval mode during a training step.")
    if bool(module.cfg.training.get("gradient_checkpointing", False)) and not checkpointing:
        raise RuntimeError(
            "training.gradient_checkpointing is true but checkpointing is not active in the LLM."
        )
    if encoder_training and not bool(module.cfg.training.get("tune_speech_encoder", False)):
        raise RuntimeError("The frozen speech encoder is in train mode during a training step.")


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
        self._accumulated_microbatch_losses = []
        self._training_modes_reported = False
        set_training_modes(self)

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
        # Stage 1 is text only: the speech projector and encoder get no gradient,
        # so they are always frozen.
        tune_llm = bool(self.cfg.training.get("tune_llm_backbone", False))
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
                param.requires_grad = False

        speech_encoder = inner_model.get_speech_encoder()
        if speech_encoder is not None:
            speech_encoder.eval()
            for param in speech_encoder.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            "Trainable parameters: "
            f"{trainable:,} / {total:,} ({100 * trainable / total:.2f}%) "
            f"[llm={tune_llm}, lora={use_lora}; speech projector/encoder frozen]"
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
        promote_trainable_params_to_fp32(self, self.cfg.training)

    def on_fit_start(self):
        # Runs after Lightning's precision plugin has converted the module.
        check_trainable_params_fp32(self, self.cfg.training)
        self._training_modes_reported = False

    def on_train_start(self):
        set_training_modes(self)

    def on_train_epoch_start(self):
        set_training_modes(self)

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
        if not self._training_modes_reported:
            report_training_modes(self)
            self._training_modes_reported = True
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
        grad_norms = [param.grad.detach().float().norm(2) for param in trainable_params]
        if not grad_norms:
            return

        weight_norms = [param.detach().float().norm(2) for param in trainable_params]

        global_grad_norm = torch.stack(grad_norms).norm(2)
        global_weight_norm = torch.stack(weight_norms).norm(2)
        global_grad_weight_ratio = global_grad_norm / global_weight_norm.clamp_min(1e-12)
        self.log(
            "grad_norm_global",
            global_grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "grad_weight_ratio_global",
            global_grad_weight_ratio,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

        lora_a_norms = []
        lora_a_weight_norms = []
        lora_b_norms = []
        lora_b_weight_norms = []
        for name, param in self.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            grad_norm = param.grad.detach().float().norm(2)
            weight_norm = param.detach().float().norm(2)
            if "lora_A" in name:
                lora_a_norms.append(grad_norm)
                lora_a_weight_norms.append(weight_norm)
            elif "lora_B" in name:
                lora_b_norms.append(grad_norm)
                lora_b_weight_norms.append(weight_norm)

        if lora_a_norms:
            lora_a_grad_norm = torch.stack(lora_a_norms).norm(2)
            lora_a_weight_norm = torch.stack(lora_a_weight_norms).norm(2)
            lora_a_grad_weight_ratio = lora_a_grad_norm / lora_a_weight_norm.clamp_min(1e-12)
            self.log(
                "grad_norm_lora_A",
                lora_a_grad_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "grad_weight_ratio_lora_A",
                lora_a_grad_weight_ratio,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )

        if lora_b_norms:
            lora_b_grad_norm = torch.stack(lora_b_norms).norm(2)
            lora_b_weight_norm = torch.stack(lora_b_weight_norms).norm(2)
            lora_b_grad_weight_ratio = lora_b_grad_norm / lora_b_weight_norm.clamp_min(1e-12)
            self.log(
                "grad_norm_lora_B",
                lora_b_grad_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "grad_weight_ratio_lora_B",
                lora_b_grad_weight_ratio,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )

    def build_optimizer(self) -> AdamW:
        return AdamW(
            [param for param in self.parameters() if param.requires_grad],
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

    def configure_optimizers(self):
        """AdamW with linear warmup (``training.warmup_ratio``) then cosine decay."""
        optimizer = self.build_optimizer()
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

@hydra.main(version_base=None, config_path="../../configs", config_name="stage_1")
def main(cfg: DictConfig):
    pl.seed_everything(int(cfg.data.seed), workers=True)

    module = BackboneTrainingModule(cfg)
    data_module = TextDataModule(cfg, module.tokenizer)
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
    finalize_fit_outputs(
        trainer, module, to_absolute_path(str(cfg.logging.output_dir)), tokenizer=module.tokenizer,
    )


if __name__ == "__main__":
    main()
