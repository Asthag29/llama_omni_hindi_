"""OmniSpeech speech+text Lightning module (LoRA + speech projector) and its batch collator.

Used by stage-2 training (``omni_speech.training.stage2``), inference
(``omni_speech.infer.inference``) and serving (``omni_speech.serve.model_worker``).
"""

from typing import Dict, List

import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from omni_speech.constants import IGNORE_INDEX
from omni_speech.train_utils import (
    load_omni_speech_checkpoint,
    model_dtype,
    optional_abs_path,
    resolve_checkpoint_path,
)
from omni_speech.model.language_model.omni_speech_llama import (
    OmniSpeechConfig,
    OmniSpeechLlamaForCausalLM,
)
from omni_speech.training.stage1 import (
    check_trainable_params_fp32,
    promote_trainable_params_to_fp32,
    report_training_modes,
    set_training_modes,
)


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


class OmniSpeechTrainingModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.tokenizer, self.model = self._load_model_and_tokenizer()
        self.speech_dtype = model_dtype(cfg.training.precision)
        self._maybe_apply_lora()
        self._configure_trainable_parameters()
        self._maybe_enable_gradient_checkpointing()
        # Promote before loading the init checkpoint so fp32 weights are copied
        # into fp32 parameters instead of being rounded through bf16.
        self._promote_trainable_params_to_fp32()
        self._maybe_load_init_checkpoint()
        self._accumulated_microbatch_losses = []
        self._training_modes_reported = False
        # LLM in train mode (from_pretrained leaves it in eval, which disables gradient
        # checkpointing); Lightning records and restores this mode around validation.
        set_training_modes(self)

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
        promote_trainable_params_to_fp32(self, self.cfg.training)

    def on_fit_start(self):
        # Runs after Lightning's precision plugin has converted the module.
        check_trainable_params_fp32(self, self.cfg.training)
        self._training_modes_reported = False

    def on_train_start(self):
        set_training_modes(self)

    def _keep_speech_encoder_eval(self):
        if bool(self.cfg.training.get("tune_speech_encoder", False)):
            return
        speech_encoder = self._get_inner_speech_model().get_speech_encoder()
        if speech_encoder is None:
            return
        speech_encoder.eval()

    def on_train_epoch_start(self):
        set_training_modes(self)
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

    def build_optimizer(self) -> AdamW:
        return AdamW(
            [param for param in self.parameters() if param.requires_grad],
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

    def _total_optimizer_steps(self) -> int:
        return max(1, int(self.trainer.estimated_stepping_batches))

    def configure_optimizers(self):
        """AdamW with linear warmup (``training.warmup_ratio``) then cosine decay."""
        optimizer = self.build_optimizer()
        total_steps = self._total_optimizer_steps()
        warmup_steps = int(total_steps * float(self.cfg.training.warmup_ratio))
        print(f"Cosine LR schedule: total_steps={total_steps}, warmup_steps={warmup_steps}")
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
