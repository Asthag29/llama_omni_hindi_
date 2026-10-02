import unittest
from pathlib import Path

import torch
from safetensors.torch import safe_open

from omni_speech.tts.indicf5 import IndicF5SpeechGenerator

CHECKPOINT = Path("models/indicf5/model.safetensors")


class IndicF5SpeechGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT.exists():
            raise unittest.SkipTest(f"{CHECKPOINT} not downloaded")
        generator = IndicF5SpeechGenerator(model_path=CHECKPOINT.parent, device="cpu")
        cls.state_dict = generator.model.state_dict()

    def test_gamma_beta_tensors_are_loaded_not_initialized(self):
        # GRN and layer-scale params init to zeros, so all-zeros means the
        # checkpoint tensor never arrived.
        keys = [k for k in self.state_dict if k.split(".")[-1] in ("gamma", "beta")]
        self.assertEqual(len(keys), 16, keys)
        for key in keys:
            self.assertGreater(torch.count_nonzero(self.state_dict[key]).item(), 0, key)

    def test_state_dict_matches_checkpoint_exactly(self):
        with safe_open(str(CHECKPOINT), framework="pt") as checkpoint:
            checkpoint_keys = set(checkpoint.keys())
        live_keys = set(self.state_dict)
        self.assertEqual(
            live_keys,
            checkpoint_keys,
            f"missing={checkpoint_keys - live_keys}, extra={live_keys - checkpoint_keys}",
        )
        self.assertEqual(len(live_keys), 447)


if __name__ == "__main__":
    unittest.main()
