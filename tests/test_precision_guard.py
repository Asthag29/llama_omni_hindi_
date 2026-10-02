"""training.precision guard: "*-true" half precisions must not silently cast the fp32 trainables.

A toy LightningModule (bf16 frozen base, fp32 trainable head) calls the real
``check_trainable_params_fp32`` from ``on_fit_start``, the hook the training
modules use, under real CPU ``pl.Trainer`` fits.
"""

import unittest

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from omni_speech.training.speech_module import OmniSpeechTrainingModule
from omni_speech.training.stage1 import BackboneTrainingModule, check_trainable_params_fp32


def training_cfg(precision, use_lora=True, tune_llm=True):
    return OmegaConf.create(
        {"precision": precision, "use_lora": use_lora, "tune_llm_backbone": tune_llm}
    )


class ToyModule(pl.LightningModule):
    def __init__(self, training):
        super().__init__()
        self.cfg = OmegaConf.create({"training": training})  # same layout as the real modules
        self.base = torch.nn.Linear(4, 4).to(torch.bfloat16).requires_grad_(False)
        self.lora = torch.nn.Linear(4, 1)  # trainable, fp32 (as after promotion)
        self.dtypes_at_fit_start = None

    def on_fit_start(self):
        self.dtypes_at_fit_start = {n: p.dtype for n, p in self.named_parameters() if p.requires_grad}
        check_trainable_params_fp32(self, self.cfg.training)

    def training_step(self, batch, batch_idx):
        hidden = self.base(batch[0].to(torch.bfloat16))
        return self.lora(hidden.to(self.lora.weight.dtype)).float().pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=1e-3)


def fit(module, precision):
    trainer = pl.Trainer(
        precision=precision, accelerator="cpu", devices=1, max_steps=2, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(module, DataLoader(TensorDataset(torch.randn(8, 4)), batch_size=4))
    return trainer


class PrecisionGuardTests(unittest.TestCase):
    def test_bf16_mixed_keeps_trainables_fp32(self):
        module = ToyModule(training_cfg("bf16-mixed"))
        trainer = fit(module, "bf16-mixed")
        self.assertEqual(trainer.global_step, 2)
        self.assertEqual(set(module.dtypes_at_fit_start.values()), {torch.float32})
        self.assertEqual(module.lora.weight.dtype, torch.float32)
        self.assertEqual(module.base.weight.dtype, torch.bfloat16)

    def test_true_half_precision_raises_a_clear_error(self):
        for precision in ("bf16-true", "16-true"):
            with self.subTest(precision=precision):
                module = ToyModule(training_cfg(precision))
                with self.assertRaises(RuntimeError) as ctx:
                    fit(module, precision)
                message = str(ctx.exception)
                self.assertIn(f"training.precision='{precision}'", message)
                self.assertIn("bf16-mixed", message)
                self.assertIn("lora.weight", message)

    def test_full_finetune_without_lora_is_exempt(self):
        module = ToyModule(training_cfg("bf16-true", use_lora=False, tune_llm=True))
        fit(module, "bf16-true")
        self.assertEqual(module.lora.weight.dtype, torch.bfloat16)

    def test_training_modules_run_the_check_on_fit_start(self):
        for cls in (BackboneTrainingModule, OmniSpeechTrainingModule):
            with self.subTest(cls=cls.__name__):
                fake = ToyModule(training_cfg("bf16-true"))
                cls.on_fit_start(fake)  # fp32 trainables: passes
                fake.lora.to(torch.bfloat16)
                with self.assertRaises(RuntimeError):
                    cls.on_fit_start(fake)


if __name__ == "__main__":
    unittest.main()
