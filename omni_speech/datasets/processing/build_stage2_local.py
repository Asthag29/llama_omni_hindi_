"""Build a local, filtered copy of the stage-2 speech data.

Streams the parquet files of the Hub dataset ``Pastaaaaa2003/Hindi-speech-instruct``
one at a time (at most ``--workers`` source files on disk), keeps only rows whose
id is in the stage-1 text file and whose audio is at most ``--max-seconds`` long,
and writes them, split into train/validation/test, under ``--output-dir``::

    <output-dir>/
      train/<source file name>.parquet
      validation/<source file name>.parquet
      test/<source file name>.parquet
      manifest.json
      index.parquet
      _state/<source file name>.json      per-source-file completion record (resume)

Audio bytes are copied untouched; durations come from the FLAC header only.
Memory is bounded independently of source file size: the audio column (stored
in the Hub files as one dictionary page of up to 1.6 GB per row group) is read
with a streaming page reader, one value at a time, keeping only kept rows.

Duplicate ids across source files are resolved when the manifest is built: the
occurrence from the lexicographically smallest source file wins; the others are
removed from the index and counts (``dropped.duplicate_id``) and their rows are
removed from the already-written output parquet (atomic rewrite).

    python -m omni_speech.datasets.processing.build_stage2_local
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import soundfile as sf

from omni_speech.datasets.splits import SOURCES, SPLITS, format_split_table, load_id_list, source_of

DEFAULT_REPO_ID = "Pastaaaaa2003/Hindi-speech-instruct"
DEFAULT_INSTRUCT_JSON = "data/instruct/hindi_instruct_conversations.json"
ROW_GROUP_SIZE = 100
READ_BATCH_ROWS = 16  # source rows decoded at a time (rows can be several MB each)
READ_BUFFER_BYTES = 4 << 20  # buffered column-chunk reads instead of whole-chunk reads
# hf_xet keeps large reconstruction buffers per download by default (~250 MB per
# concurrent file, measured); these caps halve that at the same throughput.
XET_ENV_DEFAULTS = {
    "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
    "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE": "128mb",
    "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE": "128mb",
    "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT": "256mb",
}
DROP_REASONS = ("not_in_stage1", "over_max_seconds", "unreadable_audio", "duplicate_id")
STATE_DIR = "_state"
DOWNLOAD_RETRIES = 4


class OutputLimitReached(RuntimeError):
    """Committing a file's outputs would push the output dir over ``--max-output-gb``."""


class Interrupted(RuntimeError):
    """Processing was stopped (Ctrl-C / SIGTERM / limit reached elsewhere)."""


# --------------------------------------------------------------------------- helpers


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _state_path(output_dir: Path, src_name: str) -> Path:
    return Path(output_dir) / STATE_DIR / f"{src_name}.json"


def _dir_size(path: Path, include_tmp: bool = True) -> int:
    total = 0
    if not Path(path).exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            if not include_tmp and name.endswith(".tmp"):
                continue
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except FileNotFoundError:
                pass
    return total


def _source_or_unknown(sample_id: str) -> str:
    try:
        return source_of(sample_id)
    except ValueError:
        return "unknown"


def audio_duration(audio) -> tuple[int, int]:
    """Return ``(frames, samplerate)`` from the audio header (no decode).

    ``audio`` is ``bytes`` or a seekable binary file-like object.
    """
    if isinstance(audio, (bytes, bytearray, memoryview)):
        audio = io.BytesIO(audio)
    info = sf.info(audio)
    if info.samplerate <= 0 or info.frames < 0:
        raise ValueError(f"bad header: frames={info.frames} samplerate={info.samplerate}")
    return int(info.frames), int(info.samplerate)


def _empty_split_stats() -> Dict:
    return {"rows": 0, "seconds": 0.0, "bytes": 0, "by_source": {s: 0 for s in SOURCES}}


# --------------------------------------------------------------------------- stage-1 ids


def iter_json_array_ids(path, chunk_chars: int = 1 << 22):
    """Yield ``element["id"]`` for each element of the JSON array in ``path``.

    Same accepted input as ``load_json_array_maybe_prefixed`` (text before the
    first ``[`` and after the closing ``]`` is ignored; the trailer may not
    contain ``]``) but streams the file in ``chunk_chars`` pieces and decodes one
    element at a time, so memory is one chunk plus one element.
    """
    decoder = json.JSONDecoder()
    ws = " \t\n\r"
    with open(path, encoding="utf-8") as file:
        buf = ""
        pos = 0
        eof = False

        def fill():
            nonlocal buf, pos, eof
            if pos > len(buf) // 2:  # compact consumed text
                buf, pos = buf[pos:], 0
            chunk = file.read(chunk_chars)
            if chunk:
                buf += chunk
            else:
                eof = True

        def skip_ws():
            nonlocal pos
            while True:
                while pos < len(buf) and buf[pos] in ws:
                    pos += 1
                if pos < len(buf) or eof:
                    return
                fill()

        while True:  # prefix: everything before the first "["
            idx = buf.find("[", pos)
            if idx != -1:
                pos = idx + 1
                break
            if eof:
                raise ValueError(f"Could not locate a JSON array in {path}")
            pos = len(buf)
            fill()

        first = True
        while True:
            skip_ws()
            if pos >= len(buf):
                raise ValueError(f"unterminated JSON array in {path}")
            if buf[pos] == "]":
                pos += 1
                break
            if not first:
                if buf[pos] != ",":
                    raise ValueError(f"expected ',' or ']' in {path}")
                pos += 1
                skip_ws()
            while True:
                try:
                    element, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"invalid JSON element in {path}") from None
                    fill()
                    continue
                if end >= len(buf) and not eof:  # may be truncated at the chunk edge
                    fill()
                    continue
                break
            pos = end
            first = False
            yield element["id"]

        while True:  # trailer: like rfind("]") in the reference loader, no "]" allowed
            if "]" in buf[pos:]:
                raise ValueError(f"unexpected ']' after the JSON array in {path}")
            if eof:
                return
            pos = len(buf)
            fill()


def load_stage1_ids(path) -> set:
    """The stage-1 id set; equal to ``{s["id"] for s in load_json_array_maybe_prefixed(path)}``.

    Streamed (see ``iter_json_array_ids``): a few tens of MB instead of ~0.9 GB
    for the full 300 MB JSON.
    """
    return set(iter_json_array_ids(path))


# --------------------------------------------------------------------------- streaming parquet reader
#
# The Hub files store each row group's audio column as ONE dictionary page
# (up to 1.6 GB, snappy). pyarrow must hold such a page compressed AND
# decompressed (2x its size) to read even a single row, so the audio column is
# read here with a small streaming page reader instead: page headers are parsed
# (Thrift compact protocol), snappy is decompressed incrementally, and values
# are handed out one at a time. Only kept rows' bytes are retained. Anything
# this reader does not support falls back to pyarrow for that row group.


class _Unsupported(Exception):
    """Column chunk layout the streaming reader does not handle (use pyarrow)."""


class _Compact:
    """Minimal Thrift compact-protocol struct reader (field id -> value)."""

    def __init__(self, f):
        self.f = f

    def _byte(self) -> int:
        b = self.f.read(1)
        if not b:
            raise _Unsupported("truncated thrift header")
        return b[0]

    def _varint(self) -> int:
        shift = result = 0
        while True:
            b = self._byte()
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result
            shift += 7

    def _zigzag(self) -> int:
        n = self._varint()
        return (n >> 1) ^ -(n & 1)

    def _value(self, t: int):
        if t in (1, 2):
            return t == 1
        if t == 3:
            return int.from_bytes(self.f.read(1), "little", signed=True)
        if t in (4, 5, 6):
            return self._zigzag()
        if t == 7:
            self.f.read(8)
            return None
        if t == 8:
            return self.f.read(self._varint())
        if t in (9, 10):
            header = self._byte()
            n = header >> 4
            if n == 15:
                n = self._varint()
            return [self._value(header & 0x0F) for _ in range(n)]
        if t == 11:
            n = self._varint()
            if n == 0:
                return {}
            kv = self._byte()
            return {self._value(kv >> 4): self._value(kv & 0x0F) for _ in range(n)}
        if t == 12:
            return self.struct()
        raise _Unsupported(f"thrift type {t}")

    def struct(self) -> Dict[int, object]:
        out: Dict[int, object] = {}
        last = 0
        while True:
            header = self._byte()
            if header == 0:
                return out
            delta = header >> 4
            fid = last + delta if delta else self._zigzag()
            last = fid
            out[fid] = self._value(header & 0x0F)


def _read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise _Unsupported("unexpected end of file")
    return data


_SNAPPY_WINDOW = 1 << 16  # google snappy compresses in 64 KiB blocks: offsets < 64 KiB


def _snappy_stream(f, n_compressed: int, read_size: int = 1 << 20):
    """Yield the decompressed output of a raw snappy block of ``n_compressed`` bytes."""
    remaining = n_compressed
    buf = b""
    pos = 0

    def need(n):
        nonlocal buf, pos, remaining
        while len(buf) - pos < n:
            if remaining <= 0:
                raise _Unsupported("truncated snappy data")
            chunk = _read_exact(f, min(read_size, remaining))
            remaining -= len(chunk)
            buf = buf[pos:] + chunk
            pos = 0

    # Preamble: uncompressed length (varint).
    total = shift = 0
    while True:
        need(1)
        b = buf[pos]
        pos += 1
        total |= (b & 0x7F) << shift
        if not b & 0x80:
            break
        shift += 7
    history = bytearray()
    produced = 0
    while produced < total:
        need(1)
        tag = buf[pos]
        pos += 1
        kind = tag & 3
        if kind == 0:
            length = tag >> 2
            if length >= 60:
                nb = length - 59
                need(nb)
                length = int.from_bytes(buf[pos : pos + nb], "little")
                pos += nb
            length += 1
            out_parts = []
            left = length
            while left:
                need(1)
                take = min(left, len(buf) - pos)
                out_parts.append(buf[pos : pos + take])
                pos += take
                left -= take
            piece = b"".join(out_parts) if len(out_parts) > 1 else out_parts[0]
        else:
            if kind == 1:
                need(1)
                length = 4 + ((tag >> 2) & 7)
                offset = ((tag >> 5) << 8) | buf[pos]
                pos += 1
            elif kind == 2:
                need(2)
                length = (tag >> 2) + 1
                offset = int.from_bytes(buf[pos : pos + 2], "little")
                pos += 2
            else:
                need(4)
                length = (tag >> 2) + 1
                offset = int.from_bytes(buf[pos : pos + 4], "little")
                pos += 4
            if offset == 0 or offset > len(history):
                raise _Unsupported(f"snappy copy offset {offset} outside the {len(history)}-byte window")
            start = len(history) - offset
            if offset >= length:
                piece = bytes(history[start : start + length])
            else:
                pattern = bytes(history[start:])
                piece = (pattern * (length // offset + 1))[:length]
        produced += len(piece)
        history += piece
        if len(history) > 2 * _SNAPPY_WINDOW:
            del history[: len(history) - _SNAPPY_WINDOW]
        yield piece
    if produced != total or remaining != 0 or pos != len(buf):
        raise _Unsupported("snappy length mismatch")


def _plain_stream(f, n: int, read_size: int = 1 << 20):
    while n > 0:
        chunk = _read_exact(f, min(read_size, n))
        n -= len(chunk)
        yield chunk


class _ByteReader:
    """``read(n)`` over an iterator of byte pieces."""

    def __init__(self, pieces):
        self.pieces = iter(pieces)
        self.buf = b""
        self.pos = 0

    def read(self, n: int) -> bytes:
        avail = len(self.buf) - self.pos
        if avail >= n:
            out = self.buf[self.pos : self.pos + n]
            self.pos += n
            return out
        parts = [self.buf[self.pos :]] if avail else []
        need = n - avail
        self.buf, self.pos = b"", 0
        while need > 0:
            piece = next(self.pieces, None)
            if piece is None:
                raise _Unsupported("page data ended early")
            if len(piece) > need:
                parts.append(piece[:need])
                self.buf, self.pos = piece, need
                need = 0
            else:
                parts.append(piece)
                need -= len(piece)
        return b"".join(parts)

    def read_all(self) -> bytes:
        parts = [self.buf[self.pos :]]
        parts.extend(self.pieces)
        self.buf, self.pos = b"", 0
        return b"".join(parts)

    def exhausted(self) -> bool:
        if self.pos < len(self.buf):
            return False
        for piece in self.pieces:
            if piece:
                self.buf, self.pos = piece, 0
                return False
        return True


def _decode_hybrid(data: bytes, bit_width: int, count: int) -> List[int]:
    """Parquet RLE / bit-packed hybrid decoding."""
    if bit_width == 0:
        return [0] * count
    out: List[int] = []
    pos = 0
    mask = (1 << bit_width) - 1
    byte_width = (bit_width + 7) // 8
    while len(out) < count:
        if pos >= len(data):
            raise _Unsupported("hybrid run truncated")
        header = shift = 0
        while True:
            b = data[pos]
            pos += 1
            header |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
        if header & 1:
            groups = header >> 1
            nbytes = groups * bit_width
            packed = int.from_bytes(data[pos : pos + nbytes], "little")
            pos += nbytes
            out.extend((packed >> (i * bit_width)) & mask for i in range(groups * 8))
        else:
            value = int.from_bytes(data[pos : pos + byte_width], "little")
            pos += byte_width
            out.extend([value] * (header >> 1))
    return out[:count]


def _iter_plain_byte_arrays(reader: _ByteReader, count: int):
    for _ in range(count):
        n = int.from_bytes(reader.read(4), "little")
        yield reader.read(n)


def _stream_byte_array_chunk(path, cc, num_rows: int, max_def: int, wanted, on_value) -> List[int]:
    """Stream one BYTE_ARRAY column chunk (no repetition) of ``num_rows`` rows.

    Calls ``on_value(rows, data)`` for each stored value whose rows intersect
    ``wanted`` (a set of row indices; values for other rows are skipped without
    being kept). Returns the definition level of every row (``max_def`` =
    value present).
    """
    codec = cc.compression
    if codec not in ("SNAPPY", "UNCOMPRESSED"):
        raise _Unsupported(f"codec {codec}")
    def_width = max_def.bit_length()
    start = cc.data_page_offset
    if cc.has_dictionary_page and cc.dictionary_page_offset:
        start = min(start, cc.dictionary_page_offset)
    end = start + cc.total_compressed_size

    def body(f, page_size, uncompressed_size, compressed=True):
        if codec == "SNAPPY" and compressed:
            return _snappy_stream(f, page_size)
        return _plain_stream(f, page_size)

    # Pass 1: page headers; small dictionary-index data pages are decoded here.
    pages = []
    with open(path, "rb") as f:
        f.seek(start)
        while f.tell() < end:
            header = _Compact(f).struct()
            page_type = header.get(1)
            size = header.get(3)
            body_at = f.tell()
            pages.append((page_type, header, body_at))
            f.seek(body_at + size)
        if f.tell() != end:
            raise _Unsupported("column chunk size mismatch")

    levels: List[int] = []
    dict_rows: Dict[int, List[int]] = {}
    plain_pages = []
    dict_page = None
    with open(path, "rb") as f:
        for page_type, header, body_at in pages:
            if page_type == 2:
                if dict_page is not None:
                    raise _Unsupported("two dictionary pages")
                dict_page = (header, body_at)
                continue
            if page_type == 0:
                dph = header.get(5) or {}
                n_values, encoding = dph.get(1), dph.get(2)
                if max_def and dph.get(3) != 3:
                    raise _Unsupported("definition levels not RLE")
                if encoding not in (0, 2, 8):
                    raise _Unsupported(f"encoding {encoding}")
                if encoding == 0:
                    plain_pages.append((page_type, header, body_at, len(levels)))
                    # Levels are inside the (compressed) body; read them while streaming.
                    levels.extend([None] * n_values)
                    continue
                f.seek(body_at)
                reader = _ByteReader(body(f, header[3], header[2]))
                page_levels = [0] * n_values
                if max_def:
                    lvl_len = int.from_bytes(reader.read(4), "little")
                    page_levels = _decode_hybrid(reader.read(lvl_len), def_width, n_values)
                else:
                    page_levels = [max_def] * n_values
                rest = reader.read_all()
            elif page_type == 3:
                dph = header.get(8) or {}
                n_values, encoding = dph.get(1), dph.get(4)
                if encoding not in (0, 2, 8):
                    raise _Unsupported(f"encoding {encoding}")
                if dph.get(6, 0):
                    raise _Unsupported("repetition levels")
                if encoding == 0:
                    plain_pages.append((page_type, header, body_at, len(levels)))
                    levels.extend([None] * n_values)
                    continue
                f.seek(body_at)
                def_len = dph.get(5, 0)
                lvl_bytes = _read_exact(f, def_len)
                page_levels = (
                    _decode_hybrid(lvl_bytes, def_width, n_values) if max_def else [max_def] * n_values
                )
                reader = _ByteReader(body(f, header[3] - def_len, header[2] - def_len, dph.get(7, True)))
                rest = reader.read_all()
            else:
                continue  # index pages etc.
            # Dictionary-encoded values: bit width byte + hybrid indices of present values.
            n_present = sum(1 for lv in page_levels if lv == max_def)
            indices = _decode_hybrid(rest[1:], rest[0], n_present) if n_present else []
            it = iter(indices)
            row0 = len(levels)
            for k, lv in enumerate(page_levels):
                if lv == max_def:
                    dict_rows.setdefault(next(it), []).append(row0 + k)
            levels.extend(page_levels)
        if len(levels) != num_rows:
            raise _Unsupported("value count mismatch")
        if dict_rows and dict_page is None:
            raise _Unsupported("dictionary indices without dictionary page")

        # Pass 2: stream the dictionary values (in dictionary order).
        if dict_page is not None:
            header, body_at = dict_page
            n_dict = (header.get(7) or {}).get(1, 0)
            if (header.get(7) or {}).get(2) not in (0, 2):
                raise _Unsupported("dictionary encoding")
            f.seek(body_at)
            reader = _ByteReader(body(f, header[3], header[2]))
            for k, data in enumerate(_iter_plain_byte_arrays(reader, n_dict)):
                rows = dict_rows.get(k)
                if rows and any(r in wanted for r in rows):
                    on_value(rows, data)
                del data
            if not reader.exhausted():
                raise _Unsupported("dictionary page has trailing data")

        # Pass 3: PLAIN data pages, in row order.
        for page_type, header, body_at, row0 in plain_pages:
            f.seek(body_at)
            if page_type == 0:
                n_values = header[5][1]
                reader = _ByteReader(body(f, header[3], header[2]))
                if max_def:
                    lvl_len = int.from_bytes(reader.read(4), "little")
                    page_levels = _decode_hybrid(reader.read(lvl_len), def_width, n_values)
                else:
                    page_levels = [max_def] * n_values
            else:
                dph = header[8]
                n_values = dph[1]
                def_len = dph.get(5, 0)
                lvl_bytes = _read_exact(f, def_len)
                page_levels = (
                    _decode_hybrid(lvl_bytes, def_width, n_values) if max_def else [max_def] * n_values
                )
                reader = _ByteReader(body(f, header[3] - def_len, header[2] - def_len, dph.get(7, True)))
            levels[row0 : row0 + n_values] = page_levels
            for k, lv in enumerate(page_levels):
                if lv != max_def:
                    continue
                n = int.from_bytes(reader.read(4), "little")
                row = row0 + k
                if row in wanted:
                    on_value([row], reader.read(n))
                else:
                    while n:  # skip without keeping
                        n -= len(reader.read(min(n, 1 << 20)))
            if not reader.exhausted():
                raise _Unsupported("data page has trailing data")
    return levels


def _audio_leaf(pf: pq.ParquetFile) -> Optional[int]:
    for i in range(len(pf.schema)):
        col = pf.schema.column(i)
        if col.path == "audio.bytes" and col.physical_type == "BYTE_ARRAY" and col.max_repetition_level == 0:
            return i
    return None


def _iter_units(pf: pq.ParquetFile, local_path: Path, stage1_ids, max_seconds: float, stop_event):
    """Yield ``(ids, n_chars, audio, build)`` per unit of rows (a row group, or a
    pyarrow batch on the fallback path).

    ``audio[i]`` is ``None`` (row not in stage 1: not inspected), an exception
    (unreadable), or ``(frames, samplerate)``. ``build(indices)`` returns a table
    with the source schema holding those rows; it is only valid for rows whose
    duration is within ``max_seconds``.
    """
    schema = pf.schema_arrow
    leaf = _audio_leaf(pf)
    other_cols = [pf.schema.column(i).path for i in range(len(pf.schema)) if i != leaf]
    for rg in range(pf.num_row_groups):
        if stop_event is not None and stop_event.is_set():
            raise Interrupted(str(local_path))
        unit = None
        if leaf is not None:
            try:
                unit = _streaming_unit(pf, local_path, rg, leaf, other_cols, schema, stage1_ids, max_seconds)
            except _Unsupported as exc:
                print(f"[info] {local_path.name} row group {rg}: pyarrow fallback ({exc})", flush=True)
                unit = None
        if unit is not None:
            yield unit
            continue
        for batch in pf.iter_batches(
            batch_size=READ_BATCH_ROWS, row_groups=[rg], use_threads=False
        ):
            if stop_event is not None and stop_event.is_set():
                raise Interrupted(str(local_path))
            yield _pyarrow_unit(batch, schema, stage1_ids)
            del batch


def _header_info(data) -> object:
    """(frames, samplerate) or the exception raised while reading the header."""
    try:
        if data is None or (isinstance(data, (bytes, bytearray)) and len(data) == 0):
            raise ValueError("empty audio")
        return audio_duration(data)
    except Exception as exc:  # unreadable header
        return exc


def _pyarrow_unit(batch: pa.RecordBatch, schema: pa.Schema, stage1_ids):
    ids = batch.column("id").to_pylist()
    n_chars = pc.utf8_length(batch.column("user_text")).to_pylist()
    audio_bytes = pc.struct_field(batch.column("audio"), "bytes")
    audio: List[object] = []
    for i, sample_id in enumerate(ids):
        if sample_id not in stage1_ids:
            audio.append(None)
            continue
        buf = audio_bytes[i].as_buffer()  # zero-copy view
        if buf is None or buf.size == 0:
            audio.append(ValueError("empty audio"))
        else:
            audio.append(_header_info(pa.BufferReader(buf)))
    del audio_bytes

    def build(indices: List[int]) -> pa.Table:
        return pa.Table.from_batches([batch.take(pa.array(indices, type=pa.int64()))], schema=schema)

    return ids, n_chars, audio, build


def _streaming_unit(pf, local_path, rg, leaf, other_cols, schema, stage1_ids, max_seconds):
    rg_meta = pf.metadata.row_group(rg)
    num_rows = rg_meta.num_rows
    small = pf.read_row_group(rg, columns=other_cols, use_threads=False)
    ids = small.column("id").to_pylist()
    n_chars = pc.utf8_length(small.column("user_text")).to_pylist()
    wanted = {i for i, sample_id in enumerate(ids) if sample_id in stage1_ids}
    audio: List[object] = [None] * num_rows
    keep_bytes: Dict[int, bytes] = {}

    def on_value(rows, data):
        info = _header_info(data)
        for row in rows:
            if row in wanted:
                audio[row] = info
                if not isinstance(info, Exception) and info[0] <= max_seconds * info[1]:
                    keep_bytes[row] = data

    max_def = pf.schema.column(leaf).max_definition_level
    levels = _stream_byte_array_chunk(
        local_path, rg_meta.column(leaf), num_rows, max_def, wanted, on_value
    )
    for row in wanted:
        if levels[row] != max_def:
            audio[row] = ValueError("null audio")
        elif audio[row] is None:
            raise _Unsupported(f"no value delivered for row {row}")

    audio_type = schema.field("audio").type
    small_audio = small.column("audio").combine_chunks()

    def build(indices: List[int]) -> pa.Table:
        take = pa.array(indices, type=pa.int64())
        children = []
        for field in audio_type:
            if field.name == "bytes":
                children.append(pa.array([keep_bytes[i] for i in indices], type=field.type))
            else:
                children.append(pc.struct_field(small_audio, field.name).take(take))
        audio_arr = pa.StructArray.from_arrays(children, fields=list(audio_type))
        columns = [
            audio_arr if field.name == "audio" else small.column(field.name).take(take)
            for field in schema
        ]
        return pa.Table.from_arrays(columns, schema=schema)

    return ids, n_chars, audio, build


# --------------------------------------------------------------------------- per-file filter


class _SplitWriter:
    """Buffers kept rows of one split and writes row groups of exactly ROW_GROUP_SIZE."""

    def __init__(self, path: Path, schema: pa.Schema):
        self.path = path
        self.tmp_path = path.with_name(path.name + ".tmp")
        self.schema = schema
        self.writer: Optional[pq.ParquetWriter] = None
        self.buffer: List[pa.Table] = []
        self.buffered = 0
        self.rows = 0

    def add(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        self.buffer.append(table)
        self.buffered += table.num_rows
        self.rows += table.num_rows
        if self.buffered >= ROW_GROUP_SIZE:
            self._flush(final=False)

    def _flush(self, final: bool) -> None:
        if not self.buffer:
            return
        table = pa.concat_tables(self.buffer)  # zero-copy
        n_full = (table.num_rows // ROW_GROUP_SIZE) * ROW_GROUP_SIZE
        n_write = table.num_rows if final else n_full
        if n_write == 0:
            return
        if self.writer is None:
            self.writer = pq.ParquetWriter(str(self.tmp_path), self.schema)
        self.writer.write_table(table.slice(0, n_write), row_group_size=ROW_GROUP_SIZE)
        rest = table.slice(n_write)
        self.buffer = [rest] if rest.num_rows else []
        self.buffered = rest.num_rows

    def close(self) -> Optional[Path]:
        """Finish the tmp file; returns its path, or None if no rows were kept."""
        self._flush(final=True)
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            return self.tmp_path
        return None

    def abort(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
            self.writer = None
        self.tmp_path.unlink(missing_ok=True)


def process_parquet(
    local_path,
    *,
    output_dir,
    stage1_ids: set,
    validation_ids: set,
    test_ids: set,
    max_seconds: float = 30.0,
    src_name: Optional[str] = None,
    commit_lock: Optional[threading.Lock] = None,
    max_output_bytes: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
    extra_state: Optional[Mapping] = None,
    streaming: bool = True,
) -> Optional[Dict]:
    """Filter one local source parquet into ``output_dir`` and write its state record.

    Returns the state record, or ``None`` if the state record already exists
    (resume: nothing is read or written). Output files are committed with
    ``os.replace`` from ``*.tmp`` siblings, then the state record is written.
    Raises ``OutputLimitReached`` (outputs discarded) if committing would make
    the output dir larger than ``max_output_bytes``. ``streaming=False`` forces
    the pyarrow reader (used by tests to compare both paths).
    """
    started = time.time()
    local_path = Path(local_path)
    output_dir = Path(output_dir)
    src_name = src_name or local_path.name
    state_path = _state_path(output_dir, src_name)
    if state_path.exists():
        return None
    state_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(str(local_path), buffer_size=READ_BUFFER_BYTES, pre_buffer=False)
    schema = pf.schema_arrow
    for split in SPLITS:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    writers = {split: _SplitWriter(output_dir / split / src_name, schema) for split in SPLITS}

    dropped = {reason: 0 for reason in DROP_REASONS}
    kept = {split: _empty_split_stats() for split in SPLITS}
    samples: List[Dict] = []
    seen: set = set()
    rows_total = 0
    if streaming:
        units = _iter_units(pf, local_path, stage1_ids, max_seconds, stop_event)
    else:
        units = (
            _pyarrow_unit(batch, schema, stage1_ids)
            for batch in pf.iter_batches(batch_size=READ_BATCH_ROWS, use_threads=False)
        )
    try:
        for ids, n_chars, audio, build in units:
            if stop_event is not None and stop_event.is_set():
                raise Interrupted(src_name)
            take: Dict[str, List[int]] = {split: [] for split in SPLITS}
            for i, sample_id in enumerate(ids):
                rows_total += 1
                if sample_id not in stage1_ids:
                    dropped["not_in_stage1"] += 1
                    continue
                if sample_id in seen:
                    dropped["duplicate_id"] += 1
                    continue
                info = audio[i]
                if info is None or isinstance(info, Exception):
                    dropped["unreadable_audio"] += 1
                    continue
                frames, samplerate = info
                if frames > max_seconds * samplerate:
                    dropped["over_max_seconds"] += 1
                    continue
                seen.add(sample_id)
                if sample_id in validation_ids:
                    split = "validation"
                elif sample_id in test_ids:
                    split = "test"
                else:
                    split = "train"
                duration = frames / samplerate
                source = _source_or_unknown(sample_id)
                take[split].append(i)
                stats = kept[split]
                stats["rows"] += 1
                stats["seconds"] += duration
                stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
                samples.append(
                    {
                        "id": sample_id,
                        "split": split,
                        "source": source,
                        "duration_s": duration,
                        "n_chars_user": int(n_chars[i] or 0),
                    }
                )
            for split in SPLITS:
                if take[split]:
                    writers[split].add(build(take[split]))
            del build, audio


        tmp_files = {split: writers[split].close() for split in SPLITS}
        # Commit: size check + replace under the lock so parallel workers cannot overshoot.
        lock = commit_lock if commit_lock is not None else threading.Lock()
        with lock:
            if max_output_bytes is not None:
                committed = _dir_size(output_dir, include_tmp=False)
                new = sum(p.stat().st_size for p in tmp_files.values() if p is not None)
                replaced = sum(
                    (output_dir / split / src_name).stat().st_size
                    for split in SPLITS
                    if (output_dir / split / src_name).exists()
                )
                if committed - replaced + new > max_output_bytes:
                    raise OutputLimitReached(
                        f"committing {src_name} would make {output_dir} "
                        f"{(committed - replaced + new) / 1e9:.2f} GB "
                        f"(limit {max_output_bytes / 1e9:.2f} GB)"
                    )
            output_files = {}
            for split in SPLITS:
                final = output_dir / split / src_name
                if tmp_files[split] is not None:
                    os.replace(tmp_files[split], final)
                    kept[split]["bytes"] = final.stat().st_size
                    output_files[split] = f"{split}/{src_name}"
                else:
                    final.unlink(missing_ok=True)
            state = {
                "src_file": src_name,
                "rows_total": rows_total,
                "dropped": dropped,
                "kept": kept,
                "output_files": output_files,
                "max_audio_seconds": max_seconds,
                "source_bytes": local_path.stat().st_size,
                "elapsed_s": round(time.time() - started, 3),
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "samples": samples,
            }
            if extra_state:
                state.update(extra_state)
            _atomic_write_text(state_path, json.dumps(state, ensure_ascii=False))
        return state
    finally:
        for writer in writers.values():
            writer.abort()


# --------------------------------------------------------------------------- manifest / index


def _rewrite_without_ids(path: Path, drop_ids: set) -> None:
    """Atomically rewrite ``path`` without the rows whose id is in ``drop_ids``."""
    if not path.exists():
        return
    pf = pq.ParquetFile(str(path))
    tmp = path.with_name(path.name + ".tmp")
    writer = None
    try:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            mask = pa.array([i not in drop_ids for i in table.column("id").to_pylist()])
            table = table.filter(mask)
            if table.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(str(tmp), pf.schema_arrow)
                writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
        if writer is not None:
            writer.close()
            writer = None
            os.replace(tmp, path)
        else:
            path.unlink()
    finally:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)


def load_states(output_dir) -> List[Dict]:
    state_dir = Path(output_dir) / STATE_DIR
    if not state_dir.exists():
        return []
    states = []
    for path in sorted(state_dir.glob("*.json")):
        states.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(states, key=lambda s: s["src_file"])


def build_manifest(
    output_dir,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    files_total: int,
    max_seconds: float = 30.0,
    rewrite_duplicates: bool = True,
) -> Dict:
    """Aggregate ``_state/*.json`` into ``manifest.json`` and ``index.parquet``.

    Cross-file duplicate ids keep the occurrence from the lexicographically
    smallest source file; the others are dropped from the index and counts
    and (if ``rewrite_duplicates``) removed from their output parquet.
    """
    output_dir = Path(output_dir)
    states = load_states(output_dir)
    dropped = {reason: 0 for reason in DROP_REASONS}
    owner: Dict[str, str] = {}
    for state in states:  # sorted by src_file: first occurrence wins
        for reason in DROP_REASONS:
            dropped[reason] += int(state["dropped"].get(reason, 0))
        for sample in state["samples"]:
            owner.setdefault(sample["id"], state["src_file"])

    splits = {split: _empty_split_stats() for split in SPLITS}
    file_counts = {split: 0 for split in SPLITS}
    index_rows = {k: [] for k in ("id", "split", "source", "duration_s", "n_chars_user", "src_file")}
    for state in states:
        src = state["src_file"]
        losers = {split: set() for split in SPLITS}
        for sample in state["samples"]:
            if owner[sample["id"]] != src:
                losers[sample["split"]].add(sample["id"])
                dropped["duplicate_id"] += 1
                continue
            stats = splits[sample["split"]]
            stats["rows"] += 1
            stats["seconds"] += sample["duration_s"]
            stats["by_source"][sample["source"]] = stats["by_source"].get(sample["source"], 0) + 1
            index_rows["id"].append(sample["id"])
            index_rows["split"].append(sample["split"])
            index_rows["source"].append(sample["source"])
            index_rows["duration_s"].append(float(sample["duration_s"]))
            index_rows["n_chars_user"].append(int(sample["n_chars_user"]))
            index_rows["src_file"].append(src)
        for split in SPLITS:
            if split not in state.get("output_files", {}):
                continue
            path = output_dir / split / src
            n_kept = state["kept"][split]["rows"] - len(losers[split])
            if losers[split]:
                if rewrite_duplicates:
                    _rewrite_without_ids(path, losers[split])
                size = path.stat().st_size if path.exists() else 0
            else:
                size = state["kept"][split]["bytes"]
            if n_kept > 0:
                file_counts[split] += 1
                splits[split]["bytes"] += size

    manifest_splits = {}
    for split in SPLITS:
        stats = splits[split]
        manifest_splits[split] = {
            "rows": stats["rows"],
            "files": file_counts[split],
            "hours": round(stats["seconds"] / 3600.0, 4),
            "bytes": stats["bytes"],
            "by_source": stats["by_source"],
        }
    manifest = {
        "source_repo": repo_id,
        "max_audio_seconds": float(max_seconds),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete": bool(files_total) and len(states) >= files_total,
        "files_done": len(states),
        "files_total": files_total,
        "splits": manifest_splits,
        "dropped": dropped,
    }
    index_schema = pa.schema(
        [
            ("id", pa.string()),
            ("split", pa.string()),
            ("source", pa.string()),
            ("duration_s", pa.float64()),
            ("n_chars_user", pa.int64()),
            ("src_file", pa.string()),
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    index_tmp = output_dir / "index.parquet.tmp"
    pq.write_table(pa.table(index_rows, schema=index_schema), str(index_tmp))
    os.replace(index_tmp, output_dir / "index.parquet")
    _atomic_write_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def format_summary(output_dir, manifest: Mapping) -> str:
    """Source x split table plus totals, for printing."""
    table = pq.read_table(str(Path(output_dir) / "index.parquet"), columns=["id", "split"])
    ids_by_split = {split: [] for split in SPLITS}
    for sample_id, split in zip(table.column("id").to_pylist(), table.column("split").to_pylist()):
        ids_by_split[split].append(sample_id)
    hours = sum(manifest["splits"][s]["hours"] for s in SPLITS)
    gb = sum(manifest["splits"][s]["bytes"] for s in SPLITS) / 1e9
    lines = [
        format_split_table(ids_by_split),
        f"total: {hours:.2f} h, {gb:.2f} GB in {sum(manifest['splits'][s]['files'] for s in SPLITS)} files; "
        f"files done {manifest['files_done']}/{manifest['files_total']} (complete={manifest['complete']})",
        "dropped: " + ", ".join(f"{k}={v}" for k, v in manifest["dropped"].items()),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- driver


def _download(repo_id: str, filename: str, local_dir: Path, stop_event: threading.Event) -> Path:
    from huggingface_hub import hf_hub_download

    delay = 10.0
    for attempt in range(DOWNLOAD_RETRIES + 1):
        if stop_event.is_set():
            raise Interrupted(filename)
        try:
            return Path(
                hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=str(local_dir))
            )
        except Exception as exc:
            if attempt == DOWNLOAD_RETRIES:
                raise
            print(
                f"[warn] download {filename} failed ({type(exc).__name__}: {exc}); "
                f"retry {attempt + 1}/{DOWNLOAD_RETRIES} in {delay:.0f}s",
                flush=True,
            )
            if stop_event.wait(delay):
                raise Interrupted(filename)
            delay *= 3
    raise AssertionError("unreachable")


def _handle_one(repo_path: str, args, ctx) -> Optional[Dict]:
    src_name = Path(repo_path).name
    if _state_path(args.output_dir, src_name).exists():
        return None
    work_dir = Path(args.tmp_dir) / src_name.removesuffix(".parquet")
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        local = _download(args.repo_id, repo_path, work_dir, ctx["stop"])
        download_s = time.time() - t0
        return process_parquet(
            local,
            output_dir=args.output_dir,
            stage1_ids=ctx["stage1"],
            validation_ids=ctx["validation"],
            test_ids=ctx["test"],
            max_seconds=args.max_seconds,
            src_name=src_name,
            commit_lock=ctx["lock"],
            max_output_bytes=ctx["max_bytes"],
            stop_event=ctx["stop"],
            extra_state={"repo_path": repo_path, "download_s": round(download_s, 3)},
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _list_repo_parquets(repo_id: str) -> List[str]:
    from huggingface_hub import list_repo_files

    return sorted(
        f for f in list_repo_files(repo_id, repo_type="dataset") if f.endswith(".parquet")
    )


def _select(files: Iterable[str], patterns: Optional[List[str]], limit: Optional[int]) -> List[str]:
    files = list(files)
    if patterns:
        files = [
            f
            for f in files
            if any(fnmatch.fnmatch(f, p) or fnmatch.fnmatch(Path(f).name, p) for p in patterns)
        ]
    if limit is not None:
        files = files[:limit]
    return files


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _clean_stale_tmp(output_dir: Path) -> None:
    for sub in (*SPLITS, STATE_DIR, "."):
        d = output_dir / sub
        if d.exists():
            for p in d.glob("*.tmp"):
                p.unlink(missing_ok=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--instruct-json", default=DEFAULT_INSTRUCT_JSON)
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--output-dir", default="data/speech")
    p.add_argument("--tmp-dir", default=None, help="download dir (default: fresh mkdtemp under /tmp)")
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="source files downloaded/filtered in parallel (default 2). Memory scales "
        "with workers, mainly through hf_xet download buffers (capped at 128 MB per "
        "file, 256 MB total by default) plus up to 100 buffered kept rows per split per "
        "worker. Measured peak RSS: ~0.42 GB (2 workers) / ~0.46 GB (4 workers) on "
        "0.4 GB source files, ~0.73 GB (2 workers) on the 1.1-1.6 GB files",
    )
    p.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="glob(s) restricting source files (matched against repo path or file name)",
    )
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--max-output-gb", type=float, default=20.0)
    p.add_argument("--manifest-only", action="store_true", help="rebuild manifest/index from _state")
    return p.parse_args(argv)


def main(argv=None) -> int:
    # Line-buffered output even when redirected to a log file.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    args = parse_args(argv)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / STATE_DIR).mkdir(exist_ok=True)
    _clean_stale_tmp(args.output_dir)

    prev_manifest = args.output_dir / "manifest.json"
    if args.manifest_only:
        try:
            files_total = len(_list_repo_parquets(args.repo_id))
        except Exception as exc:
            if not prev_manifest.exists():
                print(f"cannot determine files_total ({exc}) and no previous manifest", file=sys.stderr)
                return 1
            files_total = json.loads(prev_manifest.read_text())["files_total"]
        manifest = build_manifest(
            args.output_dir, repo_id=args.repo_id, files_total=files_total, max_seconds=args.max_seconds
        )
        print(format_summary(args.output_dir, manifest))
        return 0

    created_tmp = args.tmp_dir is None
    tmp_dir = Path(tempfile.mkdtemp(prefix="stage2_local_", dir="/tmp") if created_tmp else args.tmp_dir)
    home = Path.home().resolve()
    if tmp_dir.resolve() == home or home in tmp_dir.resolve().parents:
        print(f"--tmp-dir must not be under the home directory ({home})", file=sys.stderr)
        return 1
    tmp_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir = tmp_dir
    # Keep hf_xet's caches out of $HOME too (chunk cache disabled).
    os.environ["HF_XET_CACHE"] = str(tmp_dir / ".xet")
    for key, value in XET_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    print(f"loading stage-1 ids from {args.instruct_json} (streaming)", flush=True)
    stage1 = load_stage1_ids(args.instruct_json)
    validation = load_id_list(Path(args.splits_dir) / "validation_ids.txt")
    test = load_id_list(Path(args.splits_dir) / "test_ids.txt")
    if validation & test:
        print("validation and test id lists overlap", file=sys.stderr)
        return 1
    print(f"stage-1 ids: {len(stage1)} (validation {len(validation)}, test {len(test)})", flush=True)

    all_files = _list_repo_parquets(args.repo_id)
    selected = _select(all_files, args.files, args.limit_files)
    todo = [f for f in selected if not _state_path(args.output_dir, Path(f).name).exists()]
    print(
        f"repo files: {len(all_files)}; selected: {len(selected)}; already done: {len(selected) - len(todo)}; "
        f"to process: {len(todo)}; workers: {args.workers}; tmp: {tmp_dir}; output: {args.output_dir}",
        flush=True,
    )

    max_bytes = int(args.max_output_gb * 1e9)
    ctx = {
        "stage1": stage1,
        "validation": validation,
        "test": test,
        "lock": threading.Lock(),
        "stop": threading.Event(),
        "max_bytes": max_bytes,
    }

    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt

    old_term = signal.signal(signal.SIGTERM, _on_sigterm)
    exit_code = 0
    failures: List[str] = []
    started = time.time()
    n_done_run = 0
    kept_run = {split: 0 for split in SPLITS}
    already_kept = {split: 0 for split in SPLITS}
    for state in load_states(args.output_dir):
        for split in SPLITS:
            already_kept[split] += state["kept"][split]["rows"]
    done_total = len(selected) - len(todo)

    def _progress(force: bool = False) -> None:
        if not force and n_done_run % 10:
            return
        elapsed = time.time() - started
        remaining = len(todo) - n_done_run
        eta = elapsed / n_done_run * remaining if n_done_run else 0.0
        kept = ", ".join(f"{s} {already_kept[s] + kept_run[s]}" for s in SPLITS)
        print(
            f"[{done_total}/{len(selected)} files] kept: {kept} | out {_dir_size(args.output_dir) / 1e9:.2f} GB"
            f" | tmp {_dir_size(tmp_dir) / 1e9:.2f} GB | elapsed {_fmt_duration(elapsed)}"
            f" | ETA {_fmt_duration(eta)}",
            flush=True,
        )

    executor = ThreadPoolExecutor(max_workers=max(1, args.workers))
    try:
        if todo and _dir_size(args.output_dir, include_tmp=False) > max_bytes:
            raise OutputLimitReached(f"{args.output_dir} already exceeds {args.max_output_gb} GB")
        pending = {}
        queue = list(todo)
        while queue or pending:
            while queue and len(pending) < max(1, args.workers) and not ctx["stop"].is_set():
                name = queue.pop(0)
                pending[executor.submit(_handle_one, name, args, ctx)] = name
            if not pending:
                break
            finished, _ = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for fut in finished:
                name = pending.pop(fut)
                try:
                    state = fut.result()
                except OutputLimitReached:
                    ctx["stop"].set()
                    raise
                except Interrupted:
                    continue
                except Exception as exc:
                    failures.append(name)
                    print(f"[error] {name}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                n_done_run += 1
                done_total += 1
                if state is not None:
                    for split in SPLITS:
                        kept_run[split] += state["kept"][split]["rows"]
                    print(
                        f"  done {Path(name).name}: rows {state['rows_total']}, kept "
                        + "/".join(str(state["kept"][s]["rows"]) for s in SPLITS)
                        + f" (train/val/test), download {state.get('download_s', 0):.1f}s, "
                        f"total {state['elapsed_s'] + state.get('download_s', 0):.1f}s, "
                        f"{state['source_bytes'] / 1e6:.0f} MB",
                        flush=True,
                    )
                _progress(force=not queue and not pending)
    except OutputLimitReached as exc:
        print(
            f"\nSTOPPED: output size limit reached: {exc}. Raise --max-output-gb or free space; "
            "rerun the same command to resume.",
            flush=True,
        )
        exit_code = 2
    except KeyboardInterrupt:
        print("\ninterrupted; finishing in-flight files' cleanup (rerun to resume)...", flush=True)
        exit_code = 130
    finally:
        ctx["stop"].set()
        executor.shutdown(wait=True, cancel_futures=True)
        signal.signal(signal.SIGTERM, old_term)
        _clean_stale_tmp(args.output_dir)
        for child in list(tmp_dir.iterdir()) if tmp_dir.exists() else []:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        if created_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    manifest = build_manifest(
        args.output_dir, repo_id=args.repo_id, files_total=len(all_files), max_seconds=args.max_seconds
    )
    print(format_summary(args.output_dir, manifest), flush=True)
    if failures:
        print(f"{len(failures)} file(s) failed: {failures}; rerun to retry them", flush=True)
        if exit_code == 0:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
