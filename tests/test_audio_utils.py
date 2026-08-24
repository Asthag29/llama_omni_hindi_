import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from omni_speech.train_utils import load_audio_16k


class AudioUtilityTests(unittest.TestCase):
    def test_load_audio_converts_stereo_8k_to_mono_16k(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "sample.wav"
            samples = np.zeros((4000, 2), dtype=np.float32)
            sf.write(audio_path, samples, 8000)

            audio = load_audio_16k(str(audio_path))

            self.assertEqual(audio.dtype, np.float32)
            self.assertEqual(audio.ndim, 1)
            self.assertEqual(audio.shape[0], 8000)


if __name__ == "__main__":
    unittest.main()
