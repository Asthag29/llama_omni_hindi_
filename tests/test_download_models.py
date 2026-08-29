import tempfile
import unittest
from pathlib import Path

from omni_speech.datasets.downloader.download_models import (
    HINDI_ADAPTER,
    LLAMA_OMNI,
    MODELS_DIR,
    REPO_ROOT,
    WHISPER_SHA256,
    WHISPER_URL,
    sha256,
)


class DownloadModelsTests(unittest.TestCase):
    def test_destinations_match_runtime_layout(self):
        self.assertEqual(LLAMA_OMNI["destination"], MODELS_DIR / "llama")
        self.assertEqual(
            HINDI_ADAPTER["allow_patterns"],
            ["models/hindi_ckpt/stage_2/speech_text/checkpoints/last/*"],
        )
        self.assertEqual(
            MODELS_DIR / "speech_encoder" / "large-v3.pt",
            REPO_ROOT / "models" / "speech_encoder" / "large-v3.pt",
        )

    def test_upstream_revisions_are_pinned(self):
        for model in (LLAMA_OMNI, HINDI_ADAPTER):
            self.assertEqual(len(model["revision"]), 40)
        self.assertIn(WHISPER_SHA256, WHISPER_URL)
        self.assertTrue(WHISPER_URL.endswith("/large-v3.pt"))

    def test_sha256_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.bin"
            path.write_bytes(b"hindi-llama-omni")
            digest = sha256(path)
            self.assertEqual(len(digest), 64)
            self.assertEqual(digest, sha256(path))

    def test_readme_documents_setup_and_gated_indicf5(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("omni_speech/datasets/downloader/download_models.py", readme)
        self.assertIn("hf download ai4bharat/IndicF5", readme)
        self.assertIn("gated", readme.lower())


if __name__ == "__main__":
    unittest.main()
