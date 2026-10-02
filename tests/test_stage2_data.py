"""Stage-2 data pipeline: prompt, per-worker/per-rank partitioning, local mode.

CPU only, no network, no model weights. Partitioning is checked at the row-id
level through a real ``torch.utils.data.DataLoader``.
"""

import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from omni_speech.constants import DEFAULT_SPEECH_PROMPT, DEFAULT_SPEECH_TOKEN
from omni_speech.training import stage2


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
WORKER_COUNTS = (0, 1, 2, 4)

ROW_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("user_text", pa.string()),
        ("assistant_text", pa.string()),
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ]
)


def flac_bytes(seconds: float, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    audio = (0.1 * rng.standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="FLAC")
    return buffer.getvalue()


def write_parquet(path: Path, ids: list[str], seconds: list[float]) -> None:
    rows = {
        "id": ids,
        "user_text": [f"प्रश्न {row_id}" for row_id in ids],
        "assistant_text": [f"उत्तर {row_id}" for row_id in ids],
        "audio": [
            {"bytes": flac_bytes(sec, seed=i), "path": f"{row_id}.flac"}
            for i, (row_id, sec) in enumerate(zip(ids, seconds))
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows, schema=ROW_SCHEMA), path)


def build_synthetic_data_dir(
    root: Path,
    rows_per_file: dict | None = None,
    complete: bool = True,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Write a local stage-2 data dir following the build_stage2_local contract."""
    rows_per_file = rows_per_file or {
        "train": [7, 9, 13, 8, 11],
        "validation": [8, 10],
        "test": [9],
    }
    rng = np.random.default_rng(seed)
    ids: dict[str, list[str]] = {}
    splits = {}
    for split, counts in rows_per_file.items():
        ids[split] = []
        total_seconds = 0.0
        by_source: Counter = Counter()
        for file_idx, count in enumerate(counts):
            file_ids = [f"{split}-f{file_idx}-r{row}" for row in range(count)]
            seconds = [float(sec) for sec in rng.uniform(0.5, 2.0, size=count)]
            write_parquet(root / split / f"part-{file_idx:05d}.parquet", file_ids, seconds)
            ids[split].extend(file_ids)
            total_seconds += sum(seconds)
            by_source["src_a" if file_idx % 2 == 0 else "src_b"] += count
        splits[split] = {
            "rows": len(ids[split]),
            "files": len(counts),
            "hours": total_seconds / 3600,
            "by_source": dict(by_source),
        }
    manifest = {"max_audio_seconds": 30.0, "complete": complete, "splits": splits}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ids


class IdDataset(stage2.HFStreamingSpeechDataset):
    """Real streaming/partitioning/audio-guard path, but items are row ids."""

    def _row_to_item(self, row):
        if self._decode_row_audio(row) is None:
            return None
        return row["id"]


class PassTaggedIdDataset(IdDataset):
    def _iter_one_pass(self, pass_idx):
        for item in super()._iter_one_pass(pass_idx):
            yield pass_idx, item


def make_dataset(files, split, shuffle, repeat=False, rank=0, world_size=1, cls=IdDataset, seed=42):
    return cls(
        data_files=files,
        tokenizer=None,
        model_config=None,
        split_name=split,
        cache_dir=None,
        seed=seed,
        shuffle_buffer_size=16 if shuffle else 0,
        repeat=repeat,
        rank=rank,
        world_size=world_size,
    )


def collect(dataset, num_workers):
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    return list(loader)


def collect_two_passes(dataset, num_workers, rows_per_pass):
    """Items of passes 0 and 1 from an infinite (repeat=True) stream."""
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    passes = {0: [], 1: []}
    limit = rows_per_pass * 10
    for count, (pass_idx, row_id) in enumerate(loader):
        if pass_idx in passes:
            passes[pass_idx].append(row_id)
        if len(passes[0]) >= rows_per_pass and len(passes[1]) >= rows_per_pass:
            break
        if count > limit:
            break
    return passes[0], passes[1]


class SyntheticDataTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls._tmp.name) / "speech"
        cls.ids = build_synthetic_data_dir(cls.data_dir)
        cls.source = stage2.resolve_data_source(OmegaConf.create({"data_dir": str(cls.data_dir)}))
        cls.train_files = cls.source.split_files["train"]
        cls.val_files = cls.source.split_files["validation"]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def assertExactlyOnce(self, got, expected, msg=None):
        counts = Counter(got)
        duplicates = sorted(row_id for row_id, n in counts.items() if n > 1)
        missing = sorted(set(expected) - set(counts))
        extra = sorted(set(counts) - set(expected))
        self.assertEqual((duplicates, missing, extra), ([], [], []), msg)
        self.assertEqual(len(got), len(expected), msg)


class PromptTests(unittest.TestCase):
    def test_stage2_uses_the_inference_prompt(self):
        from omni_speech.infer import inference

        conversations = stage2._row_to_conversations({"id": "x", "assistant_text": "उत्तर"})
        human = conversations[0]
        self.assertEqual(human["from"], "human")
        self.assertIs(human["value"], DEFAULT_SPEECH_PROMPT)
        self.assertIs(human["value"], inference.DEFAULT_PROMPT)
        self.assertFalse(hasattr(stage2, "FIRST_TURN_PROMPT"))

    def test_prompt_has_one_leading_speech_placeholder(self):
        self.assertTrue(DEFAULT_SPEECH_PROMPT.startswith(DEFAULT_SPEECH_TOKEN + "\n"))
        self.assertEqual(DEFAULT_SPEECH_PROMPT.count(DEFAULT_SPEECH_TOKEN), 1)

    def test_preprocess_multimodal_keeps_the_prompt(self):
        from omni_speech.datasets.preprocess import preprocess_multimodal

        dataset = make_dataset([], "train", shuffle=False)
        conversations = stage2._row_to_conversations({"assistant_text": "उत्तर"})
        source = preprocess_multimodal([conversations], dataset.data_args)[0]
        self.assertEqual(source[0]["value"], DEFAULT_SPEECH_PROMPT)


class WorkerPartitionTests(SyntheticDataTestCase):
    def test_training_pass_yields_every_row_once(self):
        expected = self.ids["train"]
        for num_workers in WORKER_COUNTS:
            with self.subTest(num_workers=num_workers):
                dataset = make_dataset(self.train_files, "train", shuffle=True)
                self.assertExactlyOnce(collect(dataset, num_workers), expected)

    def test_repeating_training_stream_passes_are_complete_and_reordered(self):
        expected = self.ids["train"]
        for num_workers in WORKER_COUNTS:
            with self.subTest(num_workers=num_workers):
                dataset = make_dataset(
                    self.train_files, "train", shuffle=True, repeat=True, cls=PassTaggedIdDataset
                )
                first, second = collect_two_passes(dataset, num_workers, len(expected))
                self.assertExactlyOnce(first, expected)
                self.assertExactlyOnce(second, expected)
                self.assertNotEqual(first, second)

    def test_validation_yields_every_row_once_in_a_fixed_order(self):
        expected = self.ids["validation"]
        for num_workers in WORKER_COUNTS:
            with self.subTest(num_workers=num_workers):
                first = collect(make_dataset(self.val_files, "validation", shuffle=False), num_workers)
                second = collect(make_dataset(self.val_files, "validation", shuffle=False), num_workers)
                self.assertExactlyOnce(first, expected)
                self.assertEqual(first, second)

    def test_fewer_files_than_workers(self):
        # validation has 2 files, test has 1; with 4 workers some workers get
        # no file at all. Also run a repeating, shuffled stream over them: the
        # idle workers must stop instead of stalling the DataLoader.
        for split in ("validation", "test"):
            files = sorted(str(p) for p in (self.data_dir / split).glob("*.parquet"))
            expected = self.ids[split]
            for num_workers in (2, 4):
                with self.subTest(split=split, num_workers=num_workers, shuffle=False):
                    got = collect(make_dataset(files, split, shuffle=False), num_workers)
                    self.assertExactlyOnce(got, expected)
                with self.subTest(split=split, num_workers=num_workers, shuffle=True):
                    got = collect(make_dataset(files, split, shuffle=True), num_workers)
                    self.assertExactlyOnce(got, expected)
                with self.subTest(split=split, num_workers=num_workers, repeat=True):
                    dataset = make_dataset(
                        files, split, shuffle=True, repeat=True, cls=PassTaggedIdDataset
                    )
                    first, second = collect_two_passes(dataset, num_workers, len(expected))
                    self.assertExactlyOnce(first, expected)
                    self.assertExactlyOnce(second, expected)


class RankPartitionTests(SyntheticDataTestCase):
    """DDP: ranks see disjoint parts whose union is the whole split."""

    def _check(self, files, split, shuffle, world_size, num_workers):
        per_rank = [
            collect(
                make_dataset(files, split, shuffle=shuffle, rank=rank, world_size=world_size),
                num_workers,
            )
            for rank in range(world_size)
        ]
        for rank_ids in per_rank:
            self.assertTrue(rank_ids, "a rank received no rows")
        self.assertExactlyOnce([row_id for ids in per_rank for row_id in ids], self.ids[split])

    def test_ranks_times_workers(self):
        cases = [
            # 5 train files: shuffled; shard count vs world size varies.
            (self.train_files, "train", True),
            # 2 validation files: divisible by 2, not by 3.
            (self.val_files, "validation", False),
        ]
        for files, split, shuffle in cases:
            for world_size in (2, 3):
                for num_workers in (0, 2):
                    with self.subTest(split=split, world_size=world_size, num_workers=num_workers):
                        self._check(files, split, shuffle, world_size, num_workers)


class AudioLengthGuardTests(unittest.TestCase):
    def test_overlong_audio_is_skipped_and_counted(self):
        dataset = make_dataset([], "train", shuffle=False)
        long_row = {"id": "long", "audio": {"bytes": flac_bytes(31.0), "path": None}}
        exact_row = {"id": "exact", "audio": {"bytes": flac_bytes(30.0), "path": None}}
        self.assertIsNone(dataset._decode_row_audio(long_row))
        self.assertEqual(dataset._skipped_overlong_audio, 1)
        audio = dataset._decode_row_audio(exact_row)
        self.assertEqual(audio.shape[0], stage2.MAX_AUDIO_SAMPLES)
        self.assertEqual(dataset._skipped_overlong_audio, 1)

    def test_overlong_row_is_dropped_from_the_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "part-00000.parquet"
            write_parquet(path, ["short-0", "long", "short-1"], [1.0, 31.0, 1.5])
            dataset = make_dataset([str(path)], "train", shuffle=False)
            self.assertEqual(collect(dataset, 0), ["short-0", "short-1"])
            self.assertEqual(dataset._skipped_overlong_audio, 1)


class LocalModeTests(SyntheticDataTestCase):
    def test_counts_come_from_the_manifest(self):
        manifest = json.loads((self.data_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(self.source.mode, "local")
        self.assertEqual(self.source.train_samples, manifest["splits"]["train"]["rows"])
        self.assertEqual(self.source.validation_samples, manifest["splits"]["validation"]["rows"])
        self.assertEqual(self.source.test_samples, manifest["splits"]["test"]["rows"])
        self.assertEqual(len(self.train_files), 5)
        self.assertEqual(len(self.val_files), 2)
        table = stage2.format_split_table(self.source)
        self.assertIn("src_a", table)
        self.assertIn(str(manifest["splits"]["train"]["rows"]), table)

    def test_hub_mode_when_data_dir_is_null(self):
        source = stage2.resolve_data_source(
            OmegaConf.create({"data_dir": None, "train_samples": 105000, "validation_samples": 5720})
        )
        self.assertEqual((source.mode, source.train_samples, source.validation_samples), ("hub", 105000, 5720))

    def test_relative_data_dir_resolves_from_repo_root(self):
        self.assertEqual(stage2.resolve_repo_path("data/speech"), REPO_ROOT / "data" / "speech")

    def assertBuildError(self, data_dir):
        with self.assertRaises(stage2.LocalDataError) as ctx:
            stage2.resolve_data_source(OmegaConf.create({"data_dir": str(data_dir)}))
        self.assertIn("python -m omni_speech.datasets.processing.build_stage2_local", str(ctx.exception))

    def test_missing_directory_or_manifest_or_incomplete_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertBuildError(root / "does_not_exist")
            (root / "empty").mkdir()
            self.assertBuildError(root / "empty")
            build_synthetic_data_dir(root / "partial", {"train": [3], "validation": [2], "test": [2]}, complete=False)
            self.assertBuildError(root / "partial")

    def test_file_count_mismatch_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "speech"
            build_synthetic_data_dir(root, {"train": [3, 4], "validation": [2], "test": [2]})
            (root / "train" / "part-00001.parquet").unlink()
            self.assertBuildError(root)

    def test_default_config_uses_local_mode(self):
        with (REPO_ROOT / "configs" / "stage_2.yaml").open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["streaming"]["data_dir"], "data/speech")
        for key in ("repo_id", "train_samples", "validation_samples", "train_parquet_patterns"):
            self.assertIn(key, cfg["streaming"])


class ScheduleTests(unittest.TestCase):
    def test_schedule_from_manifest_count(self):
        # 40,019 rows / batch 2 = 20,010 microbatches; / accumulation 7 =
        # 2,859 optimizer steps per epoch; x 3 epochs = 8,577; 5 % warmup = 428.
        schedule = stage2.compute_schedule(40019, 2, 7, 3, 0.05)
        self.assertEqual(schedule["microbatches_per_epoch"], 20010)
        self.assertEqual(schedule["optimizer_steps_per_epoch"], 2859)
        self.assertEqual(schedule["total_optimizer_steps"], 8577)
        self.assertEqual(schedule["warmup_steps"], 428)
        self.assertEqual(stage2.compute_streaming_optimizer_steps(40019, 2, 7, 3), 8577)


@unittest.skipUnless((REPO_ROOT / "models" / "llama" / "tokenizer_config.json").exists(), "models/llama absent")
class RealTokenizerTests(SyntheticDataTestCase):
    def test_item_has_mel_and_the_inference_prompt(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(REPO_ROOT / "models" / "llama"), use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = 2048
        dataset = stage2.HFStreamingSpeechDataset(
            data_files=self.val_files,
            tokenizer=tokenizer,
            model_config=None,
            split_name="validation",
            cache_dir=None,
            seed=0,
            shuffle_buffer_size=0,
            repeat=False,
        )
        item = next(iter(dataset))
        self.assertEqual(tuple(item["speech"].shape), (3000, 128))
        ids = [int(t) for t in item["input_ids"] if int(t) >= 0]
        text = tokenizer.decode(ids)
        self.assertIn(DEFAULT_SPEECH_PROMPT[len("<speech>\n"):], text)
        self.assertEqual(int((item["input_ids"] == -200).sum()), 1)


if __name__ == "__main__":
    unittest.main()
