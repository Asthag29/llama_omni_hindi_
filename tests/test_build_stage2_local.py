import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from omni_speech.datasets.json_utils import load_json_array_maybe_prefixed
from omni_speech.datasets.processing import build_stage2_local as b2l
from omni_speech.datasets.processing.build_stage2_local import (
    OutputLimitReached,
    build_manifest,
    iter_json_array_ids,
    load_stage1_ids,
    process_parquet,
)

SR = 16000
HUB_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("user_text", pa.string()),
        ("assistant_text", pa.string()),
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ]
)


def _flac(n_frames: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(n_frames, dtype=np.int16), SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def _write_source(path: Path, rows, **writer_kwargs) -> None:
    table = pa.table(
        {
            "id": [r[0] for r in rows],
            "user_text": [r[1] for r in rows],
            "assistant_text": ["उत्तर " + r[0] for r in rows],
            "audio": [{"bytes": r[2], "path": f"{r[0]}.flac"} for r in rows],
        },
        schema=HUB_SCHEMA,
    )
    writer_kwargs.setdefault("row_group_size", 3)
    pq.write_table(table, str(path), **writer_kwargs)


class BuildStage2LocalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "out"
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.train_a = "flan_v2-1"
        self.val_a = "hh-rlhf-2"
        self.test_a = "0123456789abcdef0123456789abcdef"  # lm_sys
        self.exact_30 = "flan_v2-30"
        self.over_30 = "flan_v2-31"
        self.not_stage1 = "flan_v2-999"
        self.corrupt = "flan_v2-bad"
        self.stage1 = {self.train_a, self.val_a, self.test_a, self.exact_30, self.over_30, self.corrupt,
                       "flan_v2-dup", "flan_v2-b"}
        self.validation = {self.val_a}
        self.test = {self.test_a}
        self.rows_a = [
            (self.train_a, "नमस्ते", _flac(SR)),
            (self.val_a, "abc", _flac(SR // 2)),
            (self.test_a, "xyz12", _flac(2 * SR)),
            (self.exact_30, "thirty", _flac(30 * SR)),
            (self.over_30, "over", _flac(int(30.1 * SR))),
            (self.not_stage1, "nope", _flac(SR)),
            (self.corrupt, "bad", b"this is not flac"),
            ("flan_v2-dup", "dup", _flac(SR)),
        ]
        self.src_a = self.tmp / "batch_a.parquet"
        _write_source(self.src_a, self.rows_a)

    def _run(self, src, **kw):
        return process_parquet(
            src,
            output_dir=self.out,
            stage1_ids=self.stage1,
            validation_ids=self.validation,
            test_ids=self.test,
            max_seconds=30.0,
            **kw,
        )

    def test_filters_routes_and_copies_bytes(self):
        state = self._run(self.src_a)
        self.assertEqual(state["rows_total"], 8)
        self.assertEqual(
            state["dropped"],
            {"not_in_stage1": 1, "over_max_seconds": 1, "unreadable_audio": 1, "duplicate_id": 0},
        )
        self.assertEqual(state["kept"]["train"]["rows"], 3)
        self.assertEqual(state["kept"]["validation"]["rows"], 1)
        self.assertEqual(state["kept"]["test"]["rows"], 1)
        self.assertEqual(state["kept"]["train"]["by_source"]["flan_v2"], 3)
        self.assertEqual(state["kept"]["validation"]["by_source"]["hh-rlhf"], 1)
        self.assertEqual(state["kept"]["test"]["by_source"]["lm_sys"], 1)
        self.assertAlmostEqual(state["kept"]["train"]["seconds"], 32.0)

        src_table = pq.read_table(str(self.src_a))
        src_by_id = {r["id"]: r for r in src_table.to_pylist()}
        seen = {}
        for split in ("train", "validation", "test"):
            path = self.out / split / "batch_a.parquet"
            self.assertTrue(path.exists())
            pf = pq.ParquetFile(str(path))
            self.assertTrue(pf.schema_arrow.equals(src_table.schema, check_metadata=True))
            self.assertEqual(state["kept"][split]["bytes"], path.stat().st_size)
            for row in pf.read().to_pylist():
                self.assertEqual(row, src_by_id[row["id"]])  # incl. audio bytes, untouched
                seen[row["id"]] = split
        self.assertEqual(
            seen,
            {self.train_a: "train", self.exact_30: "train", "flan_v2-dup": "train",
             self.val_a: "validation", self.test_a: "test"},
        )
        self.assertNotIn(self.over_30, seen)

        samples = {s["id"]: s for s in state["samples"]}
        self.assertEqual(samples[self.exact_30]["duration_s"], 30.0)
        self.assertEqual(samples[self.train_a]["n_chars_user"], len("नमस्ते"))
        self.assertEqual(samples[self.test_a]["source"], "lm_sys")
        on_disk = json.loads((self.out / "_state" / "batch_a.parquet.json").read_text())
        self.assertEqual(on_disk["samples"], state["samples"])
        self.assertEqual(list((self.out / "train").glob("*.tmp")), [])

    def test_second_run_is_noop(self):
        self._run(self.src_a)
        before = {p: p.read_bytes() for p in self.out.rglob("*") if p.is_file()}
        mtimes = {p: p.stat().st_mtime_ns for p in before}
        self.assertIsNone(self._run(self.src_a))
        after = {p: p.read_bytes() for p in self.out.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(mtimes, {p: p.stat().st_mtime_ns for p in after})

    def test_output_limit_discards_outputs(self):
        with self.assertRaises(OutputLimitReached):
            self._run(self.src_a, max_output_bytes=10)
        self.assertFalse((self.out / "_state" / "batch_a.parquet.json").exists())
        self.assertEqual([p for p in self.out.rglob("*") if p.is_file()], [])

    def test_manifest_and_index_aggregate_two_states(self):
        self._run(self.src_a)
        src_b = self.tmp / "batch_b.parquet"
        # flan_v2-dup also appears in batch_a (smaller name) -> dropped from b; within-file dup too.
        _write_source(
            src_b,
            [("flan_v2-b", "bee", _flac(SR)), ("flan_v2-dup", "dup", _flac(SR)),
             ("flan_v2-b", "bee", _flac(SR))],
        )
        state_b = self._run(src_b)
        self.assertEqual(state_b["dropped"]["duplicate_id"], 1)
        self.assertEqual(state_b["kept"]["train"]["rows"], 2)

        manifest = build_manifest(self.out, repo_id="x/y", files_total=2)
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["files_done"], 2)
        self.assertEqual(
            manifest["dropped"],
            {"not_in_stage1": 1, "over_max_seconds": 1, "unreadable_audio": 1, "duplicate_id": 2},
        )
        train = manifest["splits"]["train"]
        self.assertEqual(train["rows"], 4)
        self.assertEqual(train["files"], 2)
        self.assertEqual(train["by_source"], {"anudesh": 0, "flan_v2": 4, "hh-rlhf": 0, "lm_sys": 0})
        self.assertAlmostEqual(train["hours"], round(33.0 / 3600, 4))
        self.assertEqual(manifest["splits"]["validation"]["rows"], 1)
        self.assertEqual(manifest["splits"]["test"]["files"], 1)
        self.assertEqual(
            train["bytes"],
            sum((self.out / "train" / n).stat().st_size for n in ("batch_a.parquet", "batch_b.parquet")),
        )

        index = pq.read_table(str(self.out / "index.parquet"))
        self.assertEqual(index.column_names, ["id", "split", "source", "duration_s", "n_chars_user", "src_file"])
        self.assertEqual(index.schema.field("duration_s").type, pa.float64())
        self.assertEqual(index.schema.field("n_chars_user").type, pa.int64())
        rows = index.to_pylist()
        self.assertEqual(len(rows), 6)
        dup = [r for r in rows if r["id"] == "flan_v2-dup"]
        self.assertEqual(len(dup), 1)
        self.assertEqual(dup[0]["src_file"], "batch_a.parquet")
        # The losing row was removed from batch_b's already-written output.
        b_ids = pq.read_table(str(self.out / "train" / "batch_b.parquet")).column("id").to_pylist()
        self.assertEqual(b_ids, ["flan_v2-b"])
        on_disk = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(on_disk["splits"], manifest["splits"])

        # Rebuilding is idempotent.
        again = build_manifest(self.out, repo_id="x/y", files_total=3)
        self.assertFalse(again["complete"])
        self.assertEqual(again["splits"], manifest["splits"])
        self.assertEqual(again["dropped"], manifest["dropped"])

    def test_output_row_groups_are_100_rows(self):
        src = self.tmp / "batch_big.parquet"
        rows = [(f"flan_v2-r{i}", "t", _flac(160)) for i in range(250)]
        _write_source(src, rows)
        stage1 = {r[0] for r in rows}
        state = process_parquet(src, output_dir=self.out, stage1_ids=stage1,
                                validation_ids=set(), test_ids=set())
        self.assertEqual(state["kept"]["train"]["rows"], 250)
        pf = pq.ParquetFile(str(self.out / "train" / "batch_big.parquet"))
        sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
        self.assertEqual(sizes, [100, 100, 50])
        self.assertEqual(pf.read().column("id").to_pylist(), [r[0] for r in rows])


class StreamingReaderTests(unittest.TestCase):
    """The streaming audio reader must give exactly what pyarrow gives."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        rng = np.random.default_rng(0)
        self.rows = []
        for i in range(40):
            if i % 13 == 5:
                audio = b"garbage"
            elif i % 11 == 7:
                audio = _flac(int(30.5 * SR))
            else:
                noise = (rng.standard_normal(SR // 4 + 37 * i) * 3000).astype(np.int16)
                buf = io.BytesIO()
                sf.write(buf, noise, SR, format="FLAC", subtype="PCM_16")
                audio = buf.getvalue()
            self.rows.append((f"flan_v2-{i}", f"text {i} ह", audio))
        self.rows.append(("flan_v2-3", "dup", _flac(SR)))  # within-file duplicate
        self.rows.append(("flan_v2-x", "not in stage1", _flac(SR)))
        self.rows.append(("flan_v2-same", "same bytes as row 0", self.rows[0][2]))
        self.stage1 = {r[0] for r in self.rows} - {"flan_v2-x"}
        self.validation = {"flan_v2-1", "flan_v2-2"}
        self.test = {"flan_v2-4"}

    def _run(self, src, out, streaming):
        state = process_parquet(
            src, output_dir=out, stage1_ids=self.stage1, validation_ids=self.validation,
            test_ids=self.test, streaming=streaming,
        )
        for key in ("elapsed_s", "finished_at"):
            state.pop(key)
        return state

    def test_streaming_matches_pyarrow(self):
        configs = {
            "dict_snappy": {},  # Hub layout: one dictionary page per row group
            "dict_snappy_rg": {"row_group_size": 7},
            "plain_snappy_pages": {"use_dictionary": False, "data_page_size": 20000},
            "plain_none_v2": {"use_dictionary": False, "compression": "none", "data_page_version": "2.0"},
            "dict_v2": {"data_page_version": "2.0"},
            "dict_fallback": {"dictionary_pagesize_limit": 30000, "data_page_size": 50000},
        }
        for name, kwargs in configs.items():
            with self.subTest(name):
                src = self.tmp / f"{name}.parquet"
                _write_source(src, self.rows, row_group_size=kwargs.pop("row_group_size", 50), **kwargs)
                with mock.patch.object(b2l, "_pyarrow_unit", side_effect=AssertionError("fallback used")):
                    streamed = self._run(src, self.tmp / f"s_{name}", streaming=True)
                reference = self._run(src, self.tmp / f"r_{name}", streaming=False)
                self.assertEqual(streamed, reference)
                self.assertEqual(streamed["dropped"]["unreadable_audio"], 3)  # rows 5, 18, 31
                self.assertEqual(streamed["dropped"]["over_max_seconds"], 2)  # rows 7, 29
                self.assertEqual(streamed["dropped"]["duplicate_id"], 1)
                for split in ("train", "validation", "test"):
                    a = self.tmp / f"s_{name}" / split / src.name
                    b = self.tmp / f"r_{name}" / split / src.name
                    ta, tb = pq.read_table(str(a)), pq.read_table(str(b))
                    self.assertTrue(ta.schema.equals(tb.schema, check_metadata=True))
                    self.assertTrue(ta.equals(tb))

    def test_snappy_stream_matches_reference(self):
        rng = np.random.default_rng(1)
        samples = [
            b"",
            b"a" * 100000,
            rng.integers(0, 256, 300000, dtype=np.uint8).tobytes(),
            (b"abcabcabd" * 20000) + rng.integers(0, 4, 70000, dtype=np.uint8).tobytes(),
        ]
        for data in samples:
            comp = pa.compress(data, codec="snappy", asbytes=True)
            out = b"".join(b2l._snappy_stream(io.BytesIO(comp), len(comp), read_size=1000))
            self.assertEqual(out, data)


class Stage1IdLoaderTests(unittest.TestCase):
    def test_matches_full_json_load(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        samples = [
            {"id": "c01d4234-8d55-51f5-b84f-0ddfd8a271b0",
             "conversations": [{"from": "human", "value": "नमस्ते [, ] {\"id\": 1}"}]},
            {"id": "flan_v2-7", "conversations": []},
            {"conversations": [{"id": "nested-not-top"}], "id": "hh-rlhf-3"},
            {"id": "flan_v2-7", "conversations": []},
            {"id": 42, "conversations": []},
        ]
        for name, body in {
            "plain.json": json.dumps(samples, ensure_ascii=False, indent=2),
            "prefixed.json": "log line\n" + json.dumps(samples, ensure_ascii=False) + "\ntrailer",
            "empty.json": "[ ]",
        }.items():
            path = tmp / name
            path.write_text(body, encoding="utf-8")
            expected = {s["id"] for s in load_json_array_maybe_prefixed(path)}
            self.assertEqual(set(iter_json_array_ids(path)), expected, name)
            self.assertEqual(load_stage1_ids(path), expected, name)
            # Tiny chunks force elements to straddle chunk boundaries.
            self.assertEqual(set(iter_json_array_ids(path, chunk_chars=7)), expected, name)

    def test_rejects_malformed(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = tmp / "bad.json"
        path.write_text('[{"id": "a"} {"id": "b"}]', encoding="utf-8")
        with self.assertRaises(ValueError):
            list(iter_json_array_ids(path))
        path.write_text('[{"id": "a"}] ]', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_stage1_ids(path)


if __name__ == "__main__":
    unittest.main()
