import tempfile
import unittest
from pathlib import Path

from omni_speech.train_utils import (
    is_safetensors_checkpoint,
    resolve_checkpoint_path,
    resolve_training_state_path,
)


class CheckpointUtilityTests(unittest.TestCase):
    def test_resolves_safetensors_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint-1"
            checkpoint.mkdir()
            (checkpoint / "speech_projector.safetensors").touch()

            self.assertTrue(is_safetensors_checkpoint(str(checkpoint)))
            self.assertEqual(
                Path(resolve_checkpoint_path(str(root))),
                checkpoint,
            )

    def test_resolves_lightning_training_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "trainer_state" / "last.ckpt"
            state.parent.mkdir()
            state.touch()

            self.assertEqual(
                Path(resolve_training_state_path(str(root))),
                state,
            )

    def test_rejects_weights_only_checkpoint_for_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "trainable.safetensors").touch()

            with self.assertRaises(ValueError):
                resolve_training_state_path(str(checkpoint))


if __name__ == "__main__":
    unittest.main()
