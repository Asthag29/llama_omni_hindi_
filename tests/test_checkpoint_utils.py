import tempfile
import unittest
from pathlib import Path

from omni_speech.train_utils import (
    is_safetensors_checkpoint,
    resolve_checkpoint_path,
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


if __name__ == "__main__":
    unittest.main()
