import unittest
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
CONFIG_NAMES = ("stage_1", "stage_2")


class ConfigValidationTests(unittest.TestCase):
    def _load(self, name):
        with (CONFIG_DIR / f"{name}.yaml").open(encoding="utf-8") as file:
            return yaml.safe_load(file)

    def test_only_the_known_configs_exist(self):
        self.assertEqual(
            sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")),
            ["speech_only", "stage_1", "stage_2"],
        )

    def test_speech_only_is_stage_2_without_an_init_checkpoint(self):
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            stage_2 = OmegaConf.to_container(compose(config_name="stage_2"))
            speech_only = OmegaConf.to_container(compose(config_name="speech_only"))

        self.assertIsNone(speech_only["model"]["init_checkpoint"])
        self.assertNotEqual(
            speech_only["logging"]["output_dir"], stage_2["logging"]["output_dir"]
        )
        for config in (stage_2, speech_only):
            config["model"].pop("init_checkpoint")
            config["logging"].pop("output_dir")
            config["logging"].pop("wandb_run_name")
        self.assertEqual(speech_only, stage_2)

    def test_training_configs_keep_required_model_paths(self):
        for name in CONFIG_NAMES:
            config = self._load(name)
            model = config["model"]
            self.assertNotIn("name", model)
            self.assertNotIn("path", model)
            for key in ("config_path", "model_base", "tokenizer_path"):
                self.assertIn(key, model)

    def test_no_streaming_or_removed_keys(self):
        removed = {
            "data": ("path", "input_type", "compute_mel_on_gpu", "prefetch_factor",
                     "persistent_workers", "cache_dir"),
            "training": ("lr_scheduler_type",),
            "logging": ("log_file", "csv_metrics_file", "checkpoint_monitor"),
        }
        for name in CONFIG_NAMES:
            config = self._load(name)
            self.assertNotIn("streaming", config, name)
            for section, keys in removed.items():
                for key in keys:
                    self.assertNotIn(key, config[section], f"{name}: {section}.{key}")

    def test_stage1_data_keys(self):
        data = self._load("stage_1")["data"]
        self.assertEqual(data["json_path"], "data/instruct/hindi_instruct_conversations.json")
        self.assertEqual(data["validation_ids_path"], "data/splits/validation_ids.txt")
        self.assertEqual(data["test_ids_path"], "data/splits/test_ids.txt")
        training = self._load("stage_1")["training"]
        for key in ("tune_speech_projector", "tune_speech_encoder"):
            self.assertNotIn(key, training)

    def test_stage2_data_keys(self):
        data = self._load("stage_2")["data"]
        self.assertEqual(data["speech_dir"], "data/speech")
        self.assertIsInstance(data["shuffle_buffer_size"], int)
        self.assertEqual(data["mel_size"], 128)

    def test_configs_parse_with_omegaconf_and_hydra(self):
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            for name in CONFIG_NAMES:
                with self.subTest(name=name):
                    plain = OmegaConf.load(CONFIG_DIR / f"{name}.yaml")
                    self.assertIsInstance(plain.training.learning_rate, float)
                    composed = compose(config_name=name)
                    OmegaConf.resolve(composed)
                    self.assertIsInstance(composed.training.learning_rate, float)
                    self.assertEqual(composed.training.precision, "bf16-mixed")

    def test_learning_rate_is_a_float_for_plain_yaml_too(self):
        for name in CONFIG_NAMES:
            self.assertIsInstance(self._load(name)["training"]["learning_rate"], float, name)

    def test_inference_root_is_repository_relative(self):
        inference = (
            REPO_ROOT / "omni_speech" / "infer" / "inference.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Path(__file__).resolve().parents[2]", inference)
        self.assertNotIn("/dss/dsshome1/", inference)


if __name__ == "__main__":
    unittest.main()
