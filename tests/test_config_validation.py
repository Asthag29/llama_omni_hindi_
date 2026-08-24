import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


class ConfigValidationTests(unittest.TestCase):
    def _load(self, name):
        with (CONFIG_DIR / name).open(encoding="utf-8") as file:
            return yaml.safe_load(file)

    def test_training_configs_keep_required_model_paths(self):
        for name in ("stage_1.yaml", "stage_2.yaml", "combined.yaml"):
            config = self._load(name)
            model = config["model"]
            self.assertNotIn("name", model)
            self.assertNotIn("path", model)
            for key in ("config_path", "model_base", "tokenizer_path"):
                self.assertIn(key, model)

    def test_stage2_does_not_keep_unused_data_path(self):
        stage2 = self._load("stage_2.yaml")
        self.assertNotIn("path", stage2["data"])

    def test_inference_root_is_repository_relative(self):
        inference = (
            REPO_ROOT / "omni_speech" / "infer" / "inference.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Path(__file__).resolve().parents[2]", inference)
        self.assertNotIn("/dss/dsshome1/", inference)


if __name__ == "__main__":
    unittest.main()
