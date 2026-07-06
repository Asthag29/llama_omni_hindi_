"""Materialize only the FLAC files needed for the current speech shard.

The sharded trainer uses this module to keep a fixed validation set plus one
train shard on local disk. It downloads HF parquet files as temporary inputs,
extracts matching FLACs, and can delete stale FLACs from the managed audio root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE


DEFAULT_REPO_ID = "Pastaaaaa2003/hindi-llama-omni"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_PARQUET_PREFIX = "data"
FIRST_TURN_PROMPT = (
    "<speech>\nकृपया उपयोगकर्ता के भाषण में प्रश्नों का सीधे उत्तर दें।"
)


def speech_filename(item: dict) -> str | None:
    speech_path = item.get("speech")
    if not speech_path:
        return None
    return os.path.basename(str(speech_path))


def filenames_from_manifest(path: str | Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        samples = json.load(f)
    return {
        filename
        for filename in (speech_filename(item) for item in samples)
        if filename
    }


def load_index(path: str | Path) -> dict[str, str]:
    index_path = Path(path)
    if not index_path.is_file():
        return {}
    with index_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return dict(payload.get("flac_to_parquet", payload))


def save_index(path: str | Path, index: dict[str, str]) -> None:
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump({"flac_to_parquet": index}, f, ensure_ascii=False, indent=2)
    tmp_path.replace(index_path)


def _repo_cache_name(repo_id: str, repo_type: str) -> str:
    prefix = {
        "dataset": "datasets",
        "model": "models",
        "space": "spaces",
    }.get(repo_type, repo_type)
    return f"{prefix}--{repo_id.replace('/', '--')}"


def clear_huggingface_repo_cache(
    repo_id: str,
    repo_type: str = DEFAULT_REPO_TYPE,
    cache_root: str | Path | None = None,
) -> int:
    """Delete the local HF hub cache for one repo and return bytes removed."""
    root = Path(cache_root) if cache_root else Path(HF_HUB_CACHE)
    cache_dir = root / _repo_cache_name(repo_id, repo_type)
    if not cache_dir.exists():
        return 0

    total_bytes = sum(
        path.stat().st_size
        for path in cache_dir.rglob("*")
        if path.is_file()
    )
    shutil.rmtree(cache_dir)
    return total_bytes


def list_parquet_files(repo_id: str, repo_type: str, parquet_prefix: str) -> list[str]:
    api = HfApi()
    items = api.list_repo_tree(repo_id, repo_type=repo_type, path_in_repo=parquet_prefix)
    return sorted(item.path for item in items if item.path.endswith(".parquet"))


def list_matching_parquet_files(
    repo_id: str,
    repo_type: str,
    parquet_prefix: str,
    patterns: Iterable[str],
) -> list[str]:
    parquets = list_parquet_files(repo_id, repo_type, parquet_prefix)
    pattern_list = list(patterns)
    return [
        parquet
        for parquet in parquets
        if any(fnmatch(parquet, pattern) for pattern in pattern_list)
    ]


def _audio_name_and_bytes(audio_value) -> tuple[str | None, bytes | None]:
    if isinstance(audio_value, dict):
        audio_path = audio_value.get("path")
        audio_bytes = audio_value.get("bytes")
    else:
        audio_path = None
        audio_bytes = None
    if not audio_path:
        return None, audio_bytes
    filename = os.path.basename(str(audio_path))
    if filename and not filename.endswith(".flac"):
        filename = f"{filename}.flac"
    return filename, audio_bytes


def _sample_from_row(sample_id: str, assistant_text: str, flac_name: str) -> dict:
    return {
        "id": sample_id,
        "speech": flac_name,
        "conversations": [
            {"from": "human", "value": FIRST_TURN_PROMPT},
            {"from": "gpt", "value": (assistant_text or "").strip()},
        ],
    }


def _download_parquet(repo_id: str, repo_type: str, parquet_path: str, cache_dir: str | None) -> tuple[str, str | None]:
    if cache_dir:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=parquet_path,
            cache_dir=cache_dir,
        )
        return local_path, None

    tmp_dir = tempfile.mkdtemp(prefix="hf_shard_parquet_")
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=parquet_path,
        cache_dir=tmp_dir,
    )
    return local_path, tmp_dir


def _run_parallel(tasks, max_workers: int, label: str, progress_every: int):
    if not tasks:
        return
    workers = max(1, min(int(max_workers), len(tasks)))
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(task) for task in tasks]
        for future in as_completed(futures):
            completed += 1
            yield future.result()
            if progress_every > 0 and (completed % progress_every == 0 or completed == len(tasks)):
                print(
                    f"{label}: processed {completed}/{len(tasks)} parquet files",
                    flush=True,
                )


def _read_manifest_samples_from_parquet(
    parquet_path: str,
    parquet_name: str,
    audio_root: Path | None,
    batch_size: int,
    write_audio: bool,
) -> tuple[list[dict], dict[str, str], int]:
    samples = []
    local_index = {}
    written_audio = 0
    parquet = pq.ParquetFile(parquet_path)
    row_offset = 0
    for record_batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["id", "assistant_text", "audio"],
    ):
        rows = record_batch.to_pydict()
        for batch_row, (sample_id, assistant_text, audio_value) in enumerate(zip(
            rows["id"],
            rows["assistant_text"],
            rows["audio"],
        )):
            flac_name, audio_bytes = _audio_name_and_bytes(audio_value)
            if not flac_name:
                continue
            local_index[flac_name] = parquet_name
            sample = _sample_from_row(sample_id, assistant_text, flac_name)
            sample["source_parquet"] = parquet_name
            sample["source_row"] = row_offset + batch_row
            samples.append(sample)
            if write_audio and audio_root is not None and audio_bytes:
                out_audio = audio_root / flac_name
                if not out_audio.is_file() or out_audio.stat().st_size == 0:
                    out_audio.write_bytes(audio_bytes)
                    written_audio += 1
        row_offset += len(rows["id"])
    return samples, local_index, written_audio


def build_manifest_from_parquets(
    out_path: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    parquet_prefix: str = DEFAULT_PARQUET_PREFIX,
    parquet_patterns: Iterable[str] = (),
    audio_root: str | Path | None = None,
    index_path: str | Path | None = None,
    cache_dir: str | None = None,
    batch_size: int = 512,
    write_audio: bool = False,
    clear_hf_cache_before_download: bool = False,
    hf_cache_root: str | Path | None = None,
    max_workers: int = 1,
    progress_every: int = 5,
) -> dict[str, int]:
    cleared_cache_bytes = 0
    if clear_hf_cache_before_download:
        cleared_cache_bytes = clear_huggingface_repo_cache(
            repo_id,
            repo_type,
            cache_root=hf_cache_root,
        )

    parquet_paths = list_matching_parquet_files(
        repo_id,
        repo_type,
        parquet_prefix,
        parquet_patterns,
    )
    if not parquet_paths:
        raise RuntimeError(
            f"No parquet files matched patterns {list(parquet_patterns)} in {repo_id}/{parquet_prefix}"
        )

    samples = []
    index = load_index(index_path) if index_path else {}
    audio_dir = Path(audio_root) if audio_root is not None else None
    if write_audio and audio_dir is not None:
        audio_dir.mkdir(parents=True, exist_ok=True)

    def task(parquet_name: str):
        local_path, tmp_dir = _download_parquet(repo_id, repo_type, parquet_name, cache_dir)
        try:
            return _read_manifest_samples_from_parquet(
                local_path,
                parquet_name,
                audio_dir,
                batch_size,
                write_audio,
            )
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    written_audio = 0
    tasks = [lambda parquet_name=parquet_name: task(parquet_name) for parquet_name in parquet_paths]
    for shard_samples, shard_index, shard_written_audio in _run_parallel(
        tasks,
        max_workers=max_workers,
        label="Manifest build",
        progress_every=progress_every,
    ):
        samples.extend(shard_samples)
        index.update(shard_index)
        written_audio += shard_written_audio

    _write_payload = samples
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_file.with_suffix(out_file.suffix + ".tmp")
    with tmp_out.open("w", encoding="utf-8") as f:
        json.dump(_write_payload, f, ensure_ascii=False, indent=2)
    tmp_out.replace(out_file)

    if index_path:
        save_index(index_path, index)

    return {
        "parquet_files": len(parquet_paths),
        "samples": len(samples),
        "audio_written": written_audio,
        "indexed": len(index),
        "cleared_hf_cache_bytes": cleared_cache_bytes,
    }


def _extract_needed_from_parquet(
    parquet_path: str,
    parquet_name: str,
    needed: set[str],
    audio_root: Path,
    index: dict[str, str],
    batch_size: int,
) -> set[str]:
    found = set()
    parquet = pq.ParquetFile(parquet_path)
    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=["audio"]):
        rows = record_batch.to_pydict()
        for audio_value in rows.get("audio", []):
            filename, audio_bytes = _audio_name_and_bytes(audio_value)
            if not filename:
                continue
            index.setdefault(filename, parquet_name)
            if filename not in needed:
                continue
            if not audio_bytes:
                continue
            out_path = audio_root / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.is_file() or out_path.stat().st_size == 0:
                out_path.write_bytes(audio_bytes)
            found.add(filename)
    return found


def _download_and_extract_needed(
    repo_id: str,
    repo_type: str,
    parquet_path: str,
    needed: set[str],
    audio_root: Path,
    cache_dir: str | None,
    batch_size: int,
) -> tuple[set[str], dict[str, str]]:
    local_path, tmp_dir = _download_parquet(repo_id, repo_type, parquet_path, cache_dir)
    local_index = {}
    found = set()
    try:
        parquet = pq.ParquetFile(local_path)
        for record_batch in parquet.iter_batches(batch_size=batch_size, columns=["audio"]):
            rows = record_batch.to_pydict()
            for audio_value in rows.get("audio", []):
                filename, audio_bytes = _audio_name_and_bytes(audio_value)
                if not filename:
                    continue
                local_index[filename] = parquet_path
                if filename not in needed or not audio_bytes:
                    continue
                out_path = audio_root / filename
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if not out_path.is_file() or out_path.stat().st_size == 0:
                    out_path.write_bytes(audio_bytes)
                found.add(filename)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return found, local_index


def _materialize_from_known_index(
    missing: set[str],
    index: dict[str, str],
    repo_id: str,
    repo_type: str,
    audio_root: Path,
    cache_dir: str | None,
    batch_size: int,
    max_workers: int,
    progress_every: int,
) -> set[str]:
    by_parquet: dict[str, set[str]] = defaultdict(set)
    for filename in missing:
        parquet_path = index.get(filename)
        if parquet_path:
            by_parquet[parquet_path].add(filename)

    found = set()
    tasks = [
        lambda parquet_path=parquet_path, filenames=filenames: _download_and_extract_needed(
            repo_id,
            repo_type,
            parquet_path,
            filenames,
            audio_root,
            cache_dir,
            batch_size,
        )
        for parquet_path, filenames in sorted(by_parquet.items())
    ]
    for just_found, local_index in _run_parallel(
        tasks,
        max_workers=max_workers,
        label="Known-index materialization",
        progress_every=progress_every,
    ):
        found.update(just_found)
        index.update(local_index)
        print(
            f"Known-index materialization: found {len(found)}/{len(missing)} files",
            flush=True,
        )
    return found


def _scan_parquet_chunk(
    parquet_paths: list[str],
    missing: set[str],
    repo_id: str,
    repo_type: str,
    audio_root: Path,
    cache_dir: str | None,
    batch_size: int,
) -> tuple[set[str], dict[str, str]]:
    found = set()
    local_index = {}
    remaining = set(missing)
    for parquet_path in parquet_paths:
        if not remaining:
            break
        just_found, shard_index = _download_and_extract_needed(
            repo_id,
            repo_type,
            parquet_path,
            remaining,
            audio_root,
            cache_dir,
            batch_size,
        )
        local_index.update(shard_index)
        found.update(just_found)
        remaining.difference_update(just_found)
    return found, local_index


def _chunks(items: list[str], n_chunks: int) -> list[list[str]]:
    n_chunks = max(1, min(n_chunks, len(items)))
    chunk_size = math.ceil(len(items) / n_chunks)
    return [
        items[start : start + chunk_size]
        for start in range(0, len(items), chunk_size)
    ]


def _partition_sizes(total: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError(f"parts must be positive, got {parts}")
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def build_sharded_manifests_from_parquets(
    out_dir: str | Path,
    num_shards: int,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    parquet_prefix: str = DEFAULT_PARQUET_PREFIX,
    parquet_patterns: Iterable[str] = (),
    index_path: str | Path | None = None,
    cache_dir: str | None = None,
    batch_size: int = 512,
    max_workers: int = 1,
    progress_every: int = 5,
) -> dict:
    parquet_paths = list_matching_parquet_files(
        repo_id,
        repo_type,
        parquet_prefix,
        parquet_patterns,
    )
    if not parquet_paths:
        raise RuntimeError(
            f"No train parquet files matched patterns {list(parquet_patterns)} in {repo_id}/{parquet_prefix}"
        )

    order = {parquet_path: idx for idx, parquet_path in enumerate(parquet_paths)}
    index = load_index(index_path) if index_path else {}
    manifest_paths = []
    sample_counts = []
    all_samples = []
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def task(parquet_name: str):
        local_path, tmp_dir = _download_parquet(repo_id, repo_type, parquet_name, cache_dir)
        try:
            shard_samples, shard_index, _ = _read_manifest_samples_from_parquet(
                local_path,
                parquet_name,
                audio_root=None,
                batch_size=batch_size,
                write_audio=False,
            )
            return parquet_name, shard_samples, shard_index
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    tasks = [lambda parquet_name=parquet_name: task(parquet_name) for parquet_name in parquet_paths]
    parquet_results = []
    for parquet_name, shard_samples, shard_index in _run_parallel(
        tasks,
        max_workers=max_workers,
        label="Train manifest build",
        progress_every=progress_every,
    ):
        parquet_results.append((parquet_name, shard_samples))
        index.update(shard_index)

    for _, shard_samples in sorted(parquet_results, key=lambda item: order[item[0]]):
        all_samples.extend(shard_samples)

    total_samples = len(all_samples)
    shard_sizes = _partition_sizes(total_samples, num_shards)
    start = 0
    shard_ranges = []
    for shard_idx, shard_size in enumerate(shard_sizes, start=1):
        end = start + shard_size
        shard_samples = all_samples[start:end]
        manifest_path = output_dir / f"train_shard_{shard_idx:02d}_of_{num_shards:02d}.json"
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(shard_samples, f, ensure_ascii=False, indent=2)
        tmp_path.replace(manifest_path)

        manifest_paths.append(str(manifest_path))
        sample_counts.append(len(shard_samples))
        if shard_samples:
            shard_ranges.append(
                {
                    "shard": shard_idx,
                    "sample_count": len(shard_samples),
                    "start_parquet": shard_samples[0].get("source_parquet"),
                    "start_row": shard_samples[0].get("source_row"),
                    "end_parquet": shard_samples[-1].get("source_parquet"),
                    "end_row": shard_samples[-1].get("source_row"),
                }
            )
        else:
            shard_ranges.append(
                {
                    "shard": shard_idx,
                    "sample_count": 0,
                    "start_parquet": None,
                    "start_row": None,
                    "end_parquet": None,
                    "end_row": None,
                }
            )
        print(
            f"Built train shard {shard_idx}/{num_shards}: "
            f"{len(shard_samples)} samples "
            f"({shard_ranges[-1]['start_parquet']}:{shard_ranges[-1]['start_row']} -> "
            f"{shard_ranges[-1]['end_parquet']}:{shard_ranges[-1]['end_row']})",
            flush=True,
        )
        start = end

    if index_path:
        save_index(index_path, index)
    return {
        "manifest_paths": manifest_paths,
        "sample_counts": sample_counts,
        "parquet_files": len(parquet_paths),
        "samples": total_samples,
        "shard_ranges": shard_ranges,
    }


def _scan_parquets_for_missing(
    missing: set[str],
    index: dict[str, str],
    repo_id: str,
    repo_type: str,
    parquet_prefix: str,
    audio_root: Path,
    cache_dir: str | None,
    batch_size: int,
    index_path: Path,
    max_workers: int,
    progress_every: int,
) -> set[str]:
    found = set()
    parquet_paths = list_parquet_files(repo_id, repo_type, parquet_prefix)
    tasks = [
        lambda chunk=chunk: _scan_parquet_chunk(
            chunk,
            missing,
            repo_id,
            repo_type,
            audio_root,
            cache_dir,
            batch_size,
        )
        for chunk in _chunks(parquet_paths, max_workers)
    ]
    for just_found, local_index in _run_parallel(
        tasks,
        max_workers=max_workers,
        label="Full-index scan",
        progress_every=progress_every,
    ):
        found.update(just_found)
        missing.difference_update(just_found)
        index.update(local_index)
        save_index(index_path, index)
        print(
            f"Full-index scan: found {len(found)} files, {len(missing)} still missing",
            flush=True,
        )
    return found


def delete_stale_flacs(audio_root: str | Path, keep_filenames: Iterable[str]) -> int:
    root = Path(audio_root)
    if not root.is_dir():
        return 0
    keep = set(keep_filenames)
    deleted = 0
    for path in root.glob("*.flac"):
        if path.name in keep:
            continue
        path.unlink()
        deleted += 1
    return deleted


def materialize_audio(
    required_filenames: Iterable[str],
    keep_filenames: Iterable[str],
    audio_root: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    parquet_prefix: str = DEFAULT_PARQUET_PREFIX,
    index_path: str | Path = "data/shards/flac_to_parquet.json",
    cache_dir: str | None = None,
    batch_size: int = 512,
    delete_extra: bool = True,
    clear_hf_cache_before_download: bool = False,
    hf_cache_root: str | Path | None = None,
    max_workers: int = 1,
    progress_every: int = 5,
) -> dict[str, int]:
    cleared_cache_bytes = 0
    if clear_hf_cache_before_download:
        cleared_cache_bytes = clear_huggingface_repo_cache(
            repo_id,
            repo_type,
            cache_root=hf_cache_root,
        )

    root = Path(audio_root)
    root.mkdir(parents=True, exist_ok=True)
    index_file = Path(index_path)
    index = load_index(index_file)

    required = set(required_filenames)
    keep = set(keep_filenames)

    if delete_extra:
        deleted = delete_stale_flacs(root, keep)
    else:
        deleted = 0

    missing = {
        filename
        for filename in required
        if not (root / filename).is_file() or (root / filename).stat().st_size == 0
    }

    found_from_index = _materialize_from_known_index(
        missing,
        index,
        repo_id,
        repo_type,
        root,
        cache_dir,
        batch_size,
        max_workers,
        progress_every,
    )
    missing.difference_update(found_from_index)

    found_from_scan = _scan_parquets_for_missing(
        missing,
        index,
        repo_id,
        repo_type,
        parquet_prefix,
        root,
        cache_dir,
        batch_size,
        index_file,
        max_workers,
        progress_every,
    )
    missing.difference_update(found_from_scan)
    save_index(index_file, index)

    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise RuntimeError(
            f"Could not materialize {len(missing)} required FLAC files from {repo_id}. "
            f"Examples: {preview}"
        )

    present = sum(1 for filename in required if (root / filename).is_file())
    return {
        "required": len(required),
        "present": present,
        "downloaded": len(found_from_index | found_from_scan),
        "deleted": deleted,
        "indexed": len(index),
        "cleared_hf_cache_bytes": cleared_cache_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--keep-manifest", action="append", default=[])
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE)
    parser.add_argument("--parquet-prefix", default=DEFAULT_PARQUET_PREFIX)
    parser.add_argument("--index-path", default="data/shards/flac_to_parquet.json")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--no-delete-extra", action="store_true")
    parser.add_argument("--clear-hf-cache-before-download", action="store_true")
    parser.add_argument("--hf-cache-root", default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    required = set()
    for manifest in args.manifest:
        required.update(filenames_from_manifest(manifest))
    keep = set(required)
    for manifest in args.keep_manifest:
        keep.update(filenames_from_manifest(manifest))

    stats = materialize_audio(
        required_filenames=required,
        keep_filenames=keep,
        audio_root=args.audio_root,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        parquet_prefix=args.parquet_prefix,
        index_path=args.index_path,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        delete_extra=not args.no_delete_extra,
        clear_hf_cache_before_download=args.clear_hf_cache_before_download,
        hf_cache_root=args.hf_cache_root,
        max_workers=args.max_workers,
        progress_every=args.progress_every,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
