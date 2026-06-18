"""Download HF parquet batches and extract FLAC + single_turn_dataset.json."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "Pastaaaaa2003/hindi-llama-omni"
INSTRUCT_REPO_ID = "Pastaaaaa2003/Hindi-speech-instruct"
FIRST_TURN_PROMPT = (
    "<speech>\nकृपया उपयोगकर्ता के भाषण में प्रश्नों का सीधे उत्तर दें।"
)


def batch_filename(batch_num: int) -> str:
    return f"data/batch_{batch_num:06d}-00000-of-00001.parquet"


def download_batch(batch_num: int, cache_dir: str | None) -> str:
    filename = batch_filename(batch_num)
    return hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        cache_dir=cache_dir,
    )


def extract_batch(
    parquet_path: str,
    flac_dir: Path,
    samples: list[dict],
    skip_existing: bool,
) -> tuple[int, int]:
    table = pq.read_table(parquet_path)
    rows = table.to_pydict()
    written = 0
    skipped = 0

    n = len(rows["id"])
    for i in range(n):
        sample_id = rows["id"][i]
        assistant_text = (rows["assistant_text"][i] or "").strip()
        audio = rows["audio"][i]
        flac_name = audio.get("path") or f"{sample_id}.flac"
        flac_path = flac_dir / flac_name

        if skip_existing and flac_path.is_file() and flac_path.stat().st_size > 0:
            skipped += 1
        else:
            audio_bytes = audio.get("bytes")
            if not audio_bytes:
                continue
            flac_path.parent.mkdir(parents=True, exist_ok=True)
            flac_path.write_bytes(audio_bytes)
            written += 1

        samples.append(
            {
                "id": f"{sample_id}_speech",
                "speech": flac_name,
                "conversations": [
                    {"from": "human", "value": FIRST_TURN_PROMPT},
                    {"from": "gpt", "value": assistant_text},
                ],
            }
        )

    return written, skipped


def download_instruct_dataset_json(out_path: Path, cache_dir: str | None) -> None:
    """Download dataset.json from Hindi-speech-instruct and save to out_path."""
    print(f"Downloading dataset.json from {INSTRUCT_REPO_ID}...")
    local = hf_hub_download(
        repo_id=INSTRUCT_REPO_ID,
        repo_type="dataset",
        filename="dataset.json",
        cache_dir=cache_dir,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(Path(local).read_bytes())
    print(f"  Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--end-batch", type=int, default=10)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/flac"),
        help="Directory for extracted FLAC files (default: data/flac)",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("data/single_turn_dataset.json"),
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--instruct-json-path",
        type=Path,
        default=Path("data/instruct_dataset.json"),
        help="Where to save dataset.json from Hindi-speech-instruct (default: data/instruct_dataset.json)",
    )
    parser.add_argument(
        "--no-instruct-json",
        dest="pull_instruct_json",
        action="store_false",
        default=True,
        help="Skip downloading dataset.json from Hindi-speech-instruct",
    )
    args = parser.parse_args()

    flac_dir = args.out_dir.resolve()
    json_path = args.json_path.resolve()
    flac_dir.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if args.pull_instruct_json:
        download_instruct_dataset_json(args.instruct_json_path.resolve(), args.cache_dir)

    samples: list[dict] = []
    total_written = 0
    total_skipped = 0

    for batch_num in range(args.start_batch, args.end_batch + 1):
        print(f"Downloading batch {batch_num:06d}...")
        parquet_path = download_batch(batch_num, args.cache_dir)
        written, skipped = extract_batch(
            parquet_path, flac_dir, samples, args.skip_existing
        )
        total_written += written
        total_skipped += skipped
        print(
            f"  batch {batch_num:06d}: {len(samples)} samples total "
            f"({written} flacs written, {skipped} skipped)"
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(
        f"Done: {len(samples)} entries -> {json_path}\n"
        f"FLAC dir: {flac_dir} ({total_written} new, {total_skipped} existing)"
    )


if __name__ == "__main__":
    main()
