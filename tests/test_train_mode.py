"""Train/eval modes under a real ``pl.Trainer`` fit (tiny CPU models).

``from_pretrained`` returns the model in eval mode and Lightning only records and
restores submodule modes, so the training modules must put the LLM in train mode
themselves: ``LlamaModel`` applies gradient checkpointing only when ``self.training``.
Uses the tiny stand-in models of ``test_trainable_precision`` (which, like
``from_pretrained``, are returned in eval mode).
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from omni_speech.infer import inference
from omni_speech.train_utils import save_omni_speech_checkpoint
from omni_speech.training.stage1 import report_training_modes
from test_trainable_precision import (
    TinyBackboneModule,
    TinyOmniSpeechModule,
    make_batch,
    make_cfg,
)

N_TRAIN_BATCHES = 4
VAL_CHECK_INTERVAL = 2  # validations after micro-batches 2 and 4 of every epoch
EPOCHS = 2
MODE_LINE = "LLM training mode: True; gradient checkpointing active: {}; speech encoder training mode: False"


def trainer_cfg(stage, gradient_checkpointing):
    cfg = make_cfg(stage=stage, gradient_checkpointing=gradient_checkpointing)
    cfg.training.warmup_ratio = 0.0
    cfg.training.lora_dropout = 0.05
    return cfg


def _recorder(base):
    class Recorder(base):
        """Records the modes seen inside every training / validation step."""

        def __init__(self, cfg):
            super().__init__(cfg)
            self.train_records = []
            self.val_records = []
            self.checkpoint_calls = 0
            inner = self._get_inner_speech_model()
            original = getattr(inner, "_gradient_checkpointing_func", None)
            if original is not None:
                def spy(*args, **kwargs):
                    self.checkpoint_calls += 1
                    return original(*args, **kwargs)

                inner._gradient_checkpointing_func = spy

        def _modes(self):
            inner = self._get_inner_speech_model()
            return {
                "model": self.model.training,
                "inner": inner.training,
                "layer0": inner.layers[0].training,
                "encoder": inner.get_speech_encoder().training,
                "epoch": self.current_epoch,
                "global_step": self.global_step,
            }

        def training_step(self, batch, batch_idx):
            before = self.checkpoint_calls
            loss = super().training_step(batch, batch_idx)
            self.train_records.append({
                **self._modes(), "batch_idx": batch_idx,
                "checkpoint_calls": self.checkpoint_calls - before,
            })
            return loss

        def validation_step(self, batch, batch_idx):
            before = self.checkpoint_calls
            loss = super().validation_step(batch, batch_idx)
            self.val_records.append({
                **self._modes(), "sanity": self.trainer.sanity_checking,
                "checkpoint_calls": self.checkpoint_calls - before,
            })
            return loss

    return Recorder


def fit(stage, gradient_checkpointing=True, eval_before_fit=False):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        module = _recorder(TinyBackboneModule if stage == 1 else TinyOmniSpeechModule)(
            trainer_cfg(stage, gradient_checkpointing)
        )
    if eval_before_fit:
        module.eval()
    batch = make_batch(with_speech=stage == 2)
    if stage == 1:
        batch = {k: v for k, v in batch.items() if k not in ("speech", "speech_lengths")}
    loader = lambda n: DataLoader([batch] * n, batch_size=None)  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        trainer = pl.Trainer(
            default_root_dir=tmp,
            accelerator="cpu",
            devices=1,
            max_epochs=EPOCHS,
            precision="bf16-mixed",
            accumulate_grad_batches=1,
            gradient_clip_val=1.8,
            logger=False,
            val_check_interval=VAL_CHECK_INTERVAL,
            num_sanity_val_steps=1,
            limit_val_batches=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        with contextlib.redirect_stdout(stdout):
            trainer.fit(module, train_dataloaders=loader(N_TRAIN_BATCHES), val_dataloaders=loader(1))
    return module, stdout.getvalue()


class TrainModeDuringFitTests(unittest.TestCase):
    def check_fit(self, stage, eval_before_fit=False):
        module, printed = fit(stage, eval_before_fit=eval_before_fit)
        records = module.train_records
        self.assertEqual(len(records), N_TRAIN_BATCHES * EPOCHS)
        n_layers = len(module._get_inner_speech_model().layers)
        for record in records:
            with self.subTest(epoch=record["epoch"], batch_idx=record["batch_idx"]):
                # First step, after the sanity check, after each mid-epoch and
                # end-of-epoch validation, and in the second epoch.
                self.assertTrue(record["model"])
                self.assertTrue(record["inner"])
                self.assertTrue(record["layer0"])
                self.assertFalse(record["encoder"])
                self.assertEqual(record["checkpoint_calls"], n_layers)
        self.assertEqual({r["epoch"] for r in records}, set(range(EPOCHS)))

        vals = module.val_records
        self.assertTrue(vals[0]["sanity"])
        self.assertEqual(sum(not v["sanity"] for v in vals), EPOCHS * N_TRAIN_BATCHES // VAL_CHECK_INTERVAL)
        for record in vals:
            self.assertFalse(record["model"])
            self.assertFalse(record["inner"])
            self.assertFalse(record["encoder"])
            self.assertEqual(record["checkpoint_calls"], 0)

        self.assertEqual(printed.count("LLM training mode:"), 1)
        self.assertIn(MODE_LINE.format(True), printed)
        self.assertIn("Gradient checkpointing enabled.", printed)

    def test_stage1(self):
        self.check_fit(stage=1)

    def test_stage2(self):
        self.check_fit(stage=2)

    def test_module_put_in_eval_before_fit_still_trains_in_train_mode(self):
        self.check_fit(stage=2, eval_before_fit=True)

    def test_without_gradient_checkpointing(self):
        module, printed = fit(stage=2, gradient_checkpointing=False)
        self.assertTrue(all(r["model"] and r["inner"] and not r["encoder"] for r in module.train_records))
        self.assertTrue(all(r["checkpoint_calls"] == 0 for r in module.train_records))
        self.assertIn(MODE_LINE.format(False), printed)

    def test_checkpointing_skipped_in_eval_mode_is_detected(self):
        # The bug being fixed: an LLM left in eval mode silently skips checkpointing.
        module = TinyBackboneModule(trainer_cfg(1, True))
        module.model.eval()
        with contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "eval mode during a training step"):
            report_training_modes(module)


class InferenceModeTests(unittest.TestCase):
    def test_loaded_module_is_in_eval_mode_and_generates(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = TinyOmniSpeechModule(make_cfg(stage=2))
            self.assertTrue(module.model.training)  # training modules start in train mode
            save_omni_speech_checkpoint(module, str(Path(tmp) / "ckpt"))
            with mock.patch.object(inference, "OmniSpeechTrainingModule", TinyOmniSpeechModule), \
                    contextlib.redirect_stdout(io.StringIO()):
                loaded, _ = inference.load_module_from_checkpoint(
                    Path(tmp) / "ckpt", make_cfg(stage=2, precision="32-true"), requested_device="cpu",
                )
        self.assertEqual({m.training for m in loaded.modules()}, {False})
        batch = make_batch()
        with torch.inference_mode():
            output = loaded.model.generate(
                batch["input_ids"][:1], speech=batch["speech"][:1],
                speech_lengths=batch["speech_lengths"][:1], max_new_tokens=3, do_sample=False,
                num_beams=1, use_cache=True, pad_token_id=0, eos_token_id=1000,
            )
        self.assertEqual(output.shape[0], 1)
        self.assertEqual({m.training for m in loaded.modules()}, {False})


if __name__ == "__main__":
    unittest.main()
