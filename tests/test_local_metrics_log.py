"""Tiny real Lightning fits that check the local train.log / metrics.csv table."""

import ast
import csv
import math
import tempfile
import unittest
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from omni_speech import train_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = ["epoch", "step", "train_loss", "train_steps_in_window", "val_loss", "lr"]


class EndlessDataset(IterableDataset):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        while True:
            x = torch.randn(4, generator=generator)
            yield x, x.sum(dim=0, keepdim=True) + 0.5 * torch.randn(1, generator=generator)


class ToyModule(pl.LightningModule):
    """Logs the same metric keys as the real training modules."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(4, 1)
        self._accumulated_microbatch_losses = []
        self.microbatch_losses_by_step = {}  # optimizer step number -> micro-batch losses
        self.lr_by_step = {}  # optimizer step number -> LR used by that step

    def _get_inner_speech_model(self):
        return self.model

    def _loss(self, batch):
        x, y = batch
        return torch.nn.functional.mse_loss(self.model(x), y)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self._accumulated_microbatch_losses.append(loss.detach())
        step = self.trainer.global_step + 1
        self.microbatch_losses_by_step.setdefault(step, []).append(float(loss.detach()))
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        return loss

    def on_before_optimizer_step(self, optimizer):
        self.lr_by_step[self.trainer.global_step + 1] = optimizer.param_groups[0]["lr"]
        if self._accumulated_microbatch_losses:
            self.log(
                "train_loss_accum",
                torch.stack(self._accumulated_microbatch_losses).mean(),
                on_step=True,
                on_epoch=False,
            )
            self._accumulated_microbatch_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1.0 + 0.1 * step))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def make_cfg(output_dir, every_n_steps=None):
    logging_cfg = {
        "output_dir": str(output_dir),
        "log_file": "logs/train.log",
        "csv": True,
        "csv_metrics_file": "csv/metrics.csv",
        "tensorboard": True,
        "wandb": False,
        "checkpoint_monitor": "val_loss",
        "save_top_k": 1,
        "save_last": False,
    }
    if every_n_steps is not None:
        logging_cfg["local_log_every_n_steps"] = every_n_steps
    return OmegaConf.create({"logging": logging_cfg})


def run_case(case, output_dir, utils=train_utils):
    """Run a tiny fit with the trainers' logger/callback helpers from ``utils``.

    Case "A": finite epochs, validation 4x per epoch, sanity check on.
    Case "B": endless stream under max_steps, validation every 20 batches.
    """
    torch.manual_seed(0)
    module = ToyModule()
    val_loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randn(8, 1)), batch_size=4)
    if case == "A":
        x = torch.randn(64, 4)
        train_loader = DataLoader(TensorDataset(x, x.sum(dim=1, keepdim=True)), batch_size=4)
        cfg = make_cfg(output_dir)
        trainer_kwargs = dict(max_epochs=2, val_check_interval=0.25, accumulate_grad_batches=2)
    elif case == "B":
        train_loader = DataLoader(EndlessDataset(), batch_size=4)
        cfg = make_cfg(output_dir, every_n_steps=4)
        trainer_kwargs = dict(max_epochs=-1, max_steps=60, val_check_interval=20, accumulate_grad_batches=2)
    else:
        raise ValueError(case)
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        accelerator="cpu",
        devices=1,
        logger=utils.build_loggers(cfg),
        callbacks=utils.build_callbacks(cfg, has_validation=True),
        log_every_n_steps=5,
        num_sanity_val_steps=2,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **trainer_kwargs,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return module, Path(output_dir)


def read_csv(output_dir):
    with open(Path(output_dir) / "csv" / "metrics.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


class LocalMetricsLogTests(unittest.TestCase):
    def _check_rows(self, module, fieldnames, rows):
        self.assertEqual(fieldnames, EXPECTED_COLUMNS)
        self.assertTrue(rows)
        previous_step = 0
        for row in rows:
            step = int(row["step"])
            self.assertGreater(step, previous_step)
            window = range(previous_step + 1, step + 1)
            self.assertEqual(int(row["train_steps_in_window"]), len(window))
            expected = sum(
                sum(module.microbatch_losses_by_step[s]) / len(module.microbatch_losses_by_step[s])
                for s in window
            ) / len(window)
            self.assertAlmostEqual(float(row["train_loss"]), expected, places=5)
            self.assertTrue(math.isclose(float(row["lr"]), module.lr_by_step[step], rel_tol=1e-9))
            int(row["epoch"])
            if row["val_loss"] != "":
                float(row["val_loss"])
            previous_step = step

    def test_case_a_finite_epochs_several_validations_per_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            module, out = run_case("A", tmp)
            fieldnames, rows = read_csv(out)
            self._check_rows(module, fieldnames, rows)
            # 16 batches/epoch, accumulate 2 -> 8 steps/epoch, validation every 2 steps.
            self.assertEqual([int(r["step"]) for r in rows], [2, 4, 6, 8, 10, 12, 14, 16])
            self.assertTrue(all(r["val_loss"] != "" for r in rows))
            self.assertNotIn(0, [int(r["step"]) for r in rows])  # no sanity-check row
            losses = [float(r["train_loss"]) for r in rows]
            for a, b in zip(losses, losses[1:]):
                self.assertNotEqual(a, b)
            log_text = (out / "logs" / "train.log").read_text(encoding="utf-8")
            for column in EXPECTED_COLUMNS:
                self.assertIn(column, log_text)
            table_rows = [line for line in log_text.splitlines() if line.startswith("| ")]
            self.assertEqual(len(table_rows), 1 + len(rows))  # header + one line per CSV row

    def test_case_b_endless_stream_under_max_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            module, out = run_case("B", tmp)
            fieldnames, rows = read_csv(out)
            self._check_rows(module, fieldnames, rows)
            self.assertEqual({int(r["epoch"]) for r in rows}, {0})  # the epoch never ends
            val_steps = [int(r["step"]) for r in rows if r["val_loss"] != ""]
            train_only_steps = [int(r["step"]) for r in rows if r["val_loss"] == ""]
            self.assertEqual(val_steps[:5], [10, 20, 30, 40, 50])
            for row in rows:
                self.assertNotEqual(row["train_loss"], "")
            expected_train_only = [s for s in range(4, 61, 4) if s not in val_steps]
            self.assertEqual(train_only_steps, expected_train_only)


class ModulesLogConsumedKeyTests(unittest.TestCase):
    """The real modules must log the per-optimizer-step key the callback reads."""

    def _logged_keys(self, path, class_name):
        tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        keys = {}
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            for node in ast.walk(method):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "log"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    keys.setdefault(node.args[0].value, set()).add(method.name)
        source = ast.get_source_segment((REPO_ROOT / path).read_text(encoding="utf-8"), cls)
        return keys, source

    def test_modules_log_train_loss_accum_and_val_loss(self):
        key = train_utils.LocalMetricsLogCallback.TRAIN_LOSS_KEY
        for path, class_name in (
            ("omni_speech/training/stage1.py", "BackboneTrainingModule"),
            ("omni_speech/training/combined.py", "OmniSpeechTrainingModule"),
        ):
            with self.subTest(class_name=class_name):
                keys, source = self._logged_keys(path, class_name)
                self.assertEqual(keys.get(key), {"on_before_optimizer_step"})
                self.assertIn("val_loss", keys)
                self.assertIn("self._accumulated_microbatch_losses.append(loss.detach())", source)


if __name__ == "__main__":
    unittest.main()
