"""Best/last weights during a fit and best_model/final_model after it (tiny CPU fits)."""

import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from omni_speech import train_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
VAL_LOSSES = [1.0, 0.8, 0.9, 0.7, 0.75]
SANITY_LOSS = 0.01  # lower than every real loss: it would win if sanity checks counted


def weight_copies(root):
    """Physical copies of the weights under ``root`` (hard links count once)."""
    inodes = set()
    for dirpath, _, filenames in os.walk(root):  # does not follow symlinked dirs
        if "trainable.safetensors" in filenames:
            st = os.stat(os.path.join(dirpath, "trainable.safetensors"))
            inodes.add((st.st_dev, st.st_ino))
    return len(inodes)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ScriptedValModule(pl.LightningModule):
    """Validation loss follows ``val_losses``; saving also writes a marker file."""

    def __init__(self, val_losses, output_root):
        super().__init__()
        self.model = torch.nn.Linear(4, 1)
        self.val_losses = list(val_losses)
        self.val_index = -1
        self.output_root = output_root
        self.copies_before_each_write = []

    def _get_inner_speech_model(self):
        return self.model

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.model(x), y)

    def on_validation_epoch_start(self):
        if not self.trainer.sanity_checking:
            self.val_index += 1

    def validation_step(self, batch, batch_idx):
        value = SANITY_LOSS if self.trainer.sanity_checking else self.val_losses[self.val_index]
        self.log("val_loss", torch.tensor(value), on_step=False, on_epoch=True, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def save_trainable_weights(self, output_dir, metadata):
        self.copies_before_each_write.append(weight_copies(self.output_root))
        train_utils.save_omni_speech_checkpoint(self, output_dir, metadata=metadata)
        marker = {
            "global_step": int(self.trainer.global_step),
            "val_index": self.val_index,
            "weight_sum": float(self.model.weight.detach().sum()),
        }
        Path(output_dir, "marker.json").write_text(json.dumps(marker), encoding="utf-8")


class Observer(pl.Callback):
    """Snapshots the output directory after every validation (runs after the checkpoint callback)."""

    def __init__(self, output_dir):
        self.out = Path(output_dir)
        self.sanity_best_exists = []
        self.snapshots = []

    def on_validation_end(self, trainer, pl_module):
        best = self.out / "checkpoints" / "best"
        if trainer.sanity_checking:
            self.sanity_best_exists.append(best.exists())
            return
        self.snapshots.append({
            "val_index": pl_module.val_index,
            "step": trainer.global_step,
            "best_marker": read_json(best / "marker.json"),
            "best_meta": read_json(best / "checkpoint_meta.json"),
            "last_marker": read_json(self.out / "checkpoints" / "last" / "marker.json"),
            "copies": weight_copies(self.out),
        })


class KillAfterValidations(pl.Callback):
    def __init__(self, n):
        self.n = n
        self.seen = 0

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self.seen += 1
            if self.seen == self.n:
                raise RuntimeError("simulated scheduler kill")


def make_cfg(output_dir, **logging_overrides):
    return OmegaConf.create({"logging": {
        "output_dir": str(output_dir),
        "log_file": "logs/train.log",
        "csv": False,
        "checkpoint_monitor": "val_loss",
        "save_last": True,
        **logging_overrides,
    }})


def run(output_dir, n_batches=10, val_check_interval=2, max_steps=-1, with_val=True,
        extra_callbacks=(), val_losses=VAL_LOSSES, cfg=None, logger=None):
    """Mimics the training mains: build_callbacks + trainer.fit + finalize_fit_outputs."""
    torch.manual_seed(0)
    cfg = make_cfg(output_dir) if cfg is None else cfg
    if logger is None:
        logger = pl.loggers.CSVLogger(str(output_dir), name="pl_logs")
    module = ScriptedValModule(val_losses, output_dir)
    x = torch.randn(n_batches, 4)
    train_loader = DataLoader(TensorDataset(x, x.sum(dim=1, keepdim=True)), batch_size=1)
    val_loader = DataLoader(TensorDataset(torch.randn(2, 4), torch.randn(2, 1)), batch_size=1)
    observer = Observer(output_dir)
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        max_steps=max_steps,
        logger=logger,
        callbacks=train_utils.build_callbacks(cfg, has_validation=with_val)
        + [observer, *extra_callbacks],
        val_check_interval=val_check_interval if with_val else None,
        num_sanity_val_steps=2,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader if with_val else None)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        train_utils.finalize_fit_outputs(trainer, module, str(output_dir))
    return module, observer, stdout.getvalue()


class BestCheckpointDuringFitTests(unittest.TestCase):
    def test_completed_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            module, observer, printed = run(out)

            # Sanity check never produced a best checkpoint.
            self.assertEqual(observer.sanity_best_exists, [False])
            # After each validation the stable path holds the best-so-far weights.
            expected_best = [(0, 1.0), (1, 0.8), (1, 0.8), (3, 0.7), (3, 0.7)]
            self.assertEqual(len(observer.snapshots), 5)
            for snap, (best_index, best_loss) in zip(observer.snapshots, expected_best):
                self.assertEqual(snap["best_marker"]["val_index"], best_index)
                self.assertAlmostEqual(snap["best_meta"]["val_loss"], best_loss, places=6)
                self.assertEqual(snap["best_meta"]["global_step"], snap["best_marker"]["global_step"])
                self.assertEqual(snap["last_marker"]["val_index"], snap["val_index"])
            # Copies on disk: 1 when latest == best, else 2 (best + last).
            self.assertEqual([s["copies"] for s in observer.snapshots], [1, 1, 2, 1, 2])
            # Never more than one existing copy while a new one is written (peak 2).
            self.assertLessEqual(max(module.copies_before_each_write), 1)

            best_step = observer.snapshots[3]["step"]
            best_meta = read_json(out / "best_model" / "checkpoint_meta.json")
            self.assertEqual(read_json(out / "best_model" / "marker.json")["val_index"], 3)
            self.assertEqual(best_meta["global_step"], best_step)
            self.assertAlmostEqual(best_meta["val_loss"], 0.7, places=6)
            self.assertIn("epoch", best_meta)

            final_marker = read_json(out / "final_model" / "marker.json")
            self.assertAlmostEqual(final_marker["weight_sum"], float(module.model.weight.sum()), places=6)
            self.assertEqual(final_marker["global_step"], module.trainer.global_step)
            final_meta = read_json(out / "final_model" / "checkpoint_meta.json")
            self.assertTrue(final_meta["final"])
            self.assertAlmostEqual(final_meta["val_loss"], 0.75, places=6)

            self.assertFalse((out / "checkpoints").exists())
            self.assertEqual(weight_copies(out), 2)
            self.assertIn(f"Best weights: step {best_step}", printed)
            self.assertIn("val_loss=0.7000", printed)
            self.assertIn(f"final weights: step {module.trainer.global_step}, val_loss=0.7500", printed)
            self.assertIn(str(out / "best_model"), printed)
            self.assertIn(str(out / "final_model"), printed)

    def test_final_step_not_validated_writes_final_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            module, observer, printed = run(out, n_batches=12, max_steps=11)
            self.assertEqual(len(observer.snapshots), 5)  # steps 2..10; step 11 not validated
            self.assertLessEqual(max(module.copies_before_each_write), 1)
            self.assertEqual(read_json(out / "best_model" / "marker.json")["val_index"], 3)
            self.assertEqual(read_json(out / "final_model" / "marker.json")["global_step"], 11)
            self.assertIsNone(read_json(out / "final_model" / "checkpoint_meta.json")["val_loss"])
            self.assertEqual(weight_copies(out), 2)
            self.assertIn("final weights: step 11, val_loss=not validated at this step", printed)

    def test_killed_mid_run_leaves_best_so_far(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "simulated scheduler kill"):
                run(out, extra_callbacks=[KillAfterValidations(3)])
            best = Path(train_utils.stable_best_checkpoint_path(str(out)))
            self.assertEqual(best, out / "checkpoints" / "best")
            self.assertTrue(train_utils.is_safetensors_checkpoint(str(best)))
            self.assertEqual(read_json(best / "marker.json")["val_index"], 1)
            self.assertAlmostEqual(read_json(best / "checkpoint_meta.json")["val_loss"], 0.8, places=6)
            self.assertEqual(read_json(out / "checkpoints" / "last" / "marker.json")["val_index"], 2)
            self.assertFalse((out / "final_model").exists())
            self.assertFalse((out / "best_model").exists())
            self.assertEqual(weight_copies(out), 2)

    def test_no_validation_best_model_is_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "checkpoints").mkdir()
            (out / "checkpoints" / "from_an_earlier_run.txt").write_text("x")
            module, observer, printed = run(out, n_batches=4, with_val=False)
            self.assertEqual(observer.snapshots, [])
            self.assertEqual(
                read_json(out / "best_model" / "marker.json"),
                read_json(out / "final_model" / "marker.json"),
            )
            self.assertTrue(read_json(out / "best_model" / "checkpoint_meta.json")["no_validation"])
            self.assertEqual(weight_copies(out), 1)  # best_model files are hard links
            self.assertIn("No validation ran: best_model = final weights", printed)
            aside = [p for p in out.iterdir() if p.name.startswith("checkpoints_previous_run_")]
            self.assertEqual(len(aside), 1)
            self.assertTrue((aside[0] / "from_an_earlier_run.txt").is_file())


class NonFiniteValidationTests(unittest.TestCase):
    """A NaN val_loss never becomes (or stays) the best once a finite value exists."""

    def test_nan_first_and_between(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            nan = float("nan")
            with self.assertLogs(level="WARNING") as logs:
                module, observer, printed = run(out, n_batches=8, val_losses=[nan, 0.9, nan, 0.8])
            self.assertEqual(len(observer.snapshots), 4)
            # Best after each validation: NaN placeholder, then 0.9, still 0.9, then 0.8.
            self.assertEqual([s["best_marker"]["val_index"] for s in observer.snapshots], [0, 1, 1, 3])
            self.assertTrue(math.isnan(observer.snapshots[0]["best_meta"]["val_loss"]))
            self.assertAlmostEqual(observer.snapshots[1]["best_meta"]["val_loss"], 0.9, places=6)
            self.assertAlmostEqual(observer.snapshots[2]["best_meta"]["val_loss"], 0.9, places=6)
            self.assertEqual(observer.snapshots[2]["last_marker"]["val_index"], 2)
            self.assertAlmostEqual(observer.snapshots[3]["best_meta"]["val_loss"], 0.8, places=6)
            self.assertEqual([s["copies"] for s in observer.snapshots], [1, 1, 2, 1])
            self.assertLessEqual(max(module.copies_before_each_write), 1)
            self.assertEqual(sum("not finite" in line for line in logs.output), 2)

            self.assertEqual(read_json(out / "best_model" / "marker.json")["val_index"], 3)
            self.assertAlmostEqual(read_json(out / "best_model" / "checkpoint_meta.json")["val_loss"], 0.8, places=6)
            self.assertIn("val_loss=0.8000", printed)
            self.assertNotIn("WARNING", printed)
            self.assertFalse((out / "checkpoints").exists())

    def test_nan_after_best_and_at_final_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            nan = float("nan")
            with self.assertLogs(level="WARNING"):
                module, observer, printed = run(out, n_batches=6, val_losses=[0.9, nan, nan])
            self.assertEqual([s["best_marker"]["val_index"] for s in observer.snapshots], [0, 0, 0])
            self.assertEqual(read_json(out / "best_model" / "marker.json")["val_index"], 0)
            final_meta = read_json(out / "final_model" / "checkpoint_meta.json")
            self.assertTrue(math.isnan(final_meta["val_loss"]))
            self.assertEqual(read_json(out / "final_model" / "marker.json")["val_index"], 2)
            self.assertIn("val_loss=0.9000", printed)
            self.assertIn(f"final weights: step {module.trainer.global_step}, val_loss=nan", printed)

    def test_all_validations_nan_keep_first_as_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            nan = float("nan")
            with self.assertLogs(level="WARNING"):
                module, observer, printed = run(out, n_batches=4, val_losses=[nan, nan])
            self.assertEqual([s["best_marker"]["val_index"] for s in observer.snapshots], [0, 0])
            self.assertEqual(read_json(out / "best_model" / "marker.json")["val_index"], 0)
            self.assertTrue(math.isnan(read_json(out / "best_model" / "checkpoint_meta.json")["val_loss"]))
            self.assertIn("val_loss=nan", printed)
            self.assertIn("WARNING: every validation val_loss was non-finite", printed)

    def test_inf_does_not_beat_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertLogs(level="WARNING"):
                run(out, n_batches=6, val_losses=[float("inf"), 1.5, float("-inf")])
            self.assertAlmostEqual(read_json(out / "best_model" / "checkpoint_meta.json")["val_loss"], 1.5, places=6)

    def test_resolve_checkpoint_path_ranks_nan_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in (("a", float("nan")), ("b", 0.9), ("c", None)):
                (root / name).mkdir()
                (root / name / "adapter_model.safetensors").touch()
                (root / name / "checkpoint_meta.json").write_text(json.dumps({"val_loss": value}))
            self.assertEqual(train_utils.resolve_checkpoint_path(str(root)), str(root / "b"))


class NoLoggerTests(unittest.TestCase):
    def test_fit_with_wandb_and_tensorboard_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cfg = make_cfg(out, wandb=False, tensorboard=False, csv=True)
            self.assertEqual(train_utils.build_loggers(cfg), [])
            callbacks = train_utils.build_callbacks(cfg, has_validation=True)
            self.assertFalse(any(isinstance(c, LearningRateMonitor) for c in callbacks))
            module, observer, printed = run(out, cfg=cfg, logger=train_utils.build_loggers(cfg))
            self.assertIsNone(module.trainer.logger)
            self.assertIn("Best weights:", printed)
            rows = (out / "csv" / "metrics.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0], "epoch,step,train_loss,train_steps_in_window,val_loss,lr")
            self.assertEqual(len(rows), 1 + len(VAL_LOSSES))
            self.assertTrue(all(row.split(",")[-1] for row in rows[1:]))  # LR recorded locally

    def test_lr_monitor_kept_when_a_logger_is_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            for overrides in ({"tensorboard": True, "wandb": False}, {"tensorboard": False, "wandb": True}):
                cfg = make_cfg(tmp, **overrides)
                callbacks = train_utils.build_callbacks(cfg, has_validation=True)
                self.assertTrue(any(isinstance(c, LearningRateMonitor) for c in callbacks), overrides)


class ConfigAndLookupTests(unittest.TestCase):
    def test_configs_init_from_best_model_without_resume_state(self):
        for name in ("stage_1.yaml", "stage_2.yaml"):
            text = (REPO_ROOT / "configs" / name).read_text(encoding="utf-8")
            self.assertNotIn("save_resume_state", text)
            if name != "stage_1.yaml":
                cfg = yaml.safe_load(text)
                self.assertEqual(cfg["model"]["init_checkpoint"], "outputs/stage_1/backbone_text/best_model")

    def _fake_checkpoint(self, path, val_loss=None):
        path.mkdir(parents=True)
        (path / "adapter_model.safetensors").touch()
        if val_loss is not None:
            (path / "checkpoint_meta.json").write_text(json.dumps({"val_loss": val_loss}))

    def test_missing_best_model_names_stable_best_path(self):
        from omni_speech.training.speech_module import OmniSpeechTrainingModule

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "stage_1"
            self._fake_checkpoint(run_dir / "checkpoints" / "step=30", 0.6974)
            os.symlink("step=30", run_dir / "checkpoints" / "best")
            fake = SimpleNamespace(cfg=OmegaConf.create(
                {"model": {"init_checkpoint": str(run_dir / "best_model")}}
            ))
            with self.assertRaises(FileNotFoundError) as ctx:
                OmniSpeechTrainingModule._maybe_load_init_checkpoint(fake)
            self.assertIn(f"pass {run_dir / 'checkpoints' / 'best'} instead", str(ctx.exception))

    def test_find_stage2_checkpoint_preference(self):
        from omni_speech.infer import inference

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "outputs" / "stage_2" / "speech_text"
            ckpts = run_dir / "checkpoints"
            self._fake_checkpoint(run_dir / "best_model", 0.5)
            self._fake_checkpoint(ckpts / "step=4", 0.6)
            os.symlink("step=4", ckpts / "best")
            self._fake_checkpoint(run_dir / "final_model", 0.9)
            self._fake_checkpoint(ckpts / "step=6", 0.9)
            os.symlink("step=6", ckpts / "last")
            self._fake_checkpoint(ckpts / "step=2", 0.7)

            with mock.patch.object(inference, "REPO_ROOT", root):
                find = inference.find_stage2_checkpoint
                self.assertEqual(find(None), run_dir / "best_model")
                os.rename(run_dir / "best_model", root / "moved_best_model")
                self.assertEqual(find(None), (ckpts / "step=4").resolve())
                os.remove(ckpts / "best")
                self.assertEqual(find(None), run_dir / "final_model")
                os.rename(run_dir / "final_model", root / "moved_final_model")
                self.assertEqual(find(None), (ckpts / "step=6").resolve())
                os.remove(ckpts / "last")
                self.assertEqual(find(None), ckpts / "step=4")  # lowest val_loss left
                self._fake_checkpoint(root / "models" / "hindi")
                self.assertEqual(find(None), root / "models" / "hindi")
                self.assertEqual(find(root / "explicit"), (root / "explicit").resolve())


if __name__ == "__main__":
    unittest.main()
