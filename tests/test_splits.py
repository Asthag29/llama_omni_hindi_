import json
import random
import unittest
import uuid
from collections import Counter
from pathlib import Path

import yaml

from omni_speech.datasets.splits import (
    SOURCES,
    load_id_list,
    partition_by_ids,
    source_of,
    stratified_split_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO_ROOT / "data" / "splits"
INSTRUCT_JSON = REPO_ROOT / "data" / "instruct" / "hindi_instruct_conversations.json"


def _synthetic_ids():
    rng = random.Random(0)
    flan = [f"flan_v2-{i}" for i in range(50)]
    lmsys = [f"{rng.getrandbits(128):032x}" for _ in range(40)]
    anudesh = [str(uuid.UUID(int=rng.getrandbits(128))) for _ in range(20)]
    hh = [f"hh-rlhf-{i}" for i in range(10)]
    return {"flan_v2": flan, "lm_sys": lmsys, "anudesh": anudesh, "hh-rlhf": hh}


class PartitionByIdsTests(unittest.TestCase):
    def test_routes_ids_and_preserves_order(self):
        samples = [{"id": name, "n": i} for i, name in enumerate("abcdefg")]
        parts = partition_by_ids(samples, {"b", "f"}, {"c", "a"})
        self.assertEqual([s["id"] for s in parts["train"]], ["d", "e", "g"])
        self.assertEqual([s["id"] for s in parts["validation"]], ["b", "f"])
        self.assertEqual([s["id"] for s in parts["test"]], ["a", "c"])
        self.assertIs(parts["train"][0], samples[3])

    def test_overlap_raises(self):
        with self.assertRaises(ValueError):
            partition_by_ids([{"id": "a"}], {"a", "b"}, {"b"})


class SourceOfTests(unittest.TestCase):
    def test_each_id_format(self):
        self.assertEqual(source_of("flan_v2-12345"), "flan_v2")
        self.assertEqual(source_of("hh-rlhf-77"), "hh-rlhf")
        self.assertEqual(source_of("c01d4234-8d55-51f5-b84f-0ddfd8a271b0"), "anudesh")
        self.assertEqual(source_of("0123456789abcdef0123456789abcdef"), "lm_sys")

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            source_of("something-else")


class StratifiedSplitIdsTests(unittest.TestCase):
    def setUp(self):
        self.by_source = _synthetic_ids()
        self.ids = [i for ids in self.by_source.values() for i in ids]

    def test_counts_disjoint_and_cover(self):
        result = stratified_split_ids(self.ids)
        train, val, test = (set(result[k]) for k in ("train", "validation", "test"))
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(train | val | test, set(self.ids))
        for split in ("train", "validation", "test"):
            self.assertEqual(result[split], sorted(result[split]))
        for source, ids in self.by_source.items():
            n = len(ids)
            n_val = sum(source_of(i) == source for i in val)
            n_test = sum(source_of(i) == source for i in test)
            self.assertEqual(n_val, round(0.1 * n), source)
            self.assertEqual(n_test, round(0.1 * n), source)
            self.assertEqual(n - n_val - n_test, sum(source_of(i) == source for i in train))

    def test_independent_of_order_and_deterministic(self):
        first = stratified_split_ids(self.ids)
        shuffled = list(self.ids)
        random.Random(123).shuffle(shuffled)
        self.assertEqual(stratified_split_ids(shuffled), first)
        self.assertEqual(stratified_split_ids(self.ids + self.ids[:5]), first)
        self.assertEqual(stratified_split_ids(self.ids), first)

    def test_seed_changes_assignment(self):
        self.assertNotEqual(
            stratified_split_ids(self.ids, seed=42),
            stratified_split_ids(self.ids, seed=7),
        )


class TrackedSplitFileTests(unittest.TestCase):
    def test_id_lists_are_disjoint_and_non_empty(self):
        val = load_id_list(SPLIT_DIR / "validation_ids.txt")
        test = load_id_list(SPLIT_DIR / "test_ids.txt")
        self.assertTrue(val)
        self.assertTrue(test)
        self.assertFalse(val & test)

    def test_id_lists_are_sorted_unique_with_trailing_newline(self):
        for name in ("validation_ids.txt", "test_ids.txt"):
            text = (SPLIT_DIR / name).read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"), name)
            lines = text.splitlines()
            self.assertEqual(lines, sorted(set(lines)), name)

    def test_stage1_config_points_at_tracked_lists(self):
        with (REPO_ROOT / "configs" / "stage_1.yaml").open(encoding="utf-8") as file:
            data = yaml.safe_load(file)["data"]
        self.assertNotIn("validation_split", data)
        self.assertTrue((REPO_ROOT / data["validation_ids_path"]).is_file())
        self.assertTrue((REPO_ROOT / data["test_ids_path"]).is_file())


@unittest.skipUnless(INSTRUCT_JSON.is_file(), f"{INSTRUCT_JSON} not present")
class RealDataSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with INSTRUCT_JSON.open(encoding="utf-8") as file:
            cls.samples = json.load(file)
        cls.val_ids = load_id_list(SPLIT_DIR / "validation_ids.txt")
        cls.test_ids = load_id_list(SPLIT_DIR / "test_ids.txt")

    def test_regenerating_reproduces_tracked_lists(self):
        result = stratified_split_ids([s["id"] for s in self.samples], seed=42)
        self.assertEqual(set(result["validation"]), self.val_ids)
        self.assertEqual(set(result["test"]), self.test_ids)

    def test_partition_of_real_json(self):
        parts = partition_by_ids(self.samples, self.val_ids, self.test_ids)
        self.assertEqual(sum(len(p) for p in parts.values()), len(self.samples))
        seen = Counter(id(s) for p in parts.values() for s in p)
        self.assertEqual(len(seen), len(self.samples))
        self.assertEqual(set(seen.values()), {1})

        totals = Counter(source_of(s["id"]) for s in self.samples)
        for split in ("train", "validation", "test"):
            counts = Counter(source_of(s["id"]) for s in parts[split])
            for source in SOURCES:
                self.assertGreater(counts[source], 0, (split, source))
                if split != "train":
                    share = 100 * counts[source] / totals[source]
                    self.assertLessEqual(abs(share - 10.0), 0.1, (split, source, share))


if __name__ == "__main__":
    unittest.main()
