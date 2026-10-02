"""Make the fixed 80/10/10 train/validation/test split of the Hindi instruct data.

The split is stratified by source (anudesh, flan_v2, hh-rlhf, lm_sys) and
depends only on the set of ids and the seed. It writes the held-out id lists
``validation_ids.txt`` and ``test_ids.txt``; every other id is train.

    python -m omni_speech.datasets.processing.split_hindi_instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omni_speech.datasets.json_utils import load_json_array_maybe_prefixed
from omni_speech.datasets.splits import (
    SPLITS,
    format_split_table,
    partition_by_ids,
    stratified_split_ids,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/instruct/hindi_instruct_conversations.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Also write train.json, validation.json and test.json next to --input.",
    )
    args = parser.parse_args()

    samples = load_json_array_maybe_prefixed(args.input.resolve())
    ids = [sample["id"] for sample in samples]
    duplicates = len(ids) - len(set(ids))
    print(f"Loaded {len(samples)} samples ({len(set(ids))} unique ids, {duplicates} duplicates) from {args.input}")

    split_ids = stratified_split_ids(ids, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("validation", "test"):
        out_path = args.output_dir / f"{name}_ids.txt"
        out_path.write_text("".join(f"{i}\n" for i in split_ids[name]), encoding="utf-8")
        print(f"Wrote {len(split_ids[name])} ids to {out_path}")

    parts = partition_by_ids(samples, split_ids["validation"], split_ids["test"])
    print(format_split_table({name: [s["id"] for s in parts[name]] for name in SPLITS}))

    if args.write_json:
        for name in SPLITS:
            out_path = args.input.parent / f"{name}.json"
            with out_path.open("w", encoding="utf-8") as file:
                json.dump(parts[name], file, ensure_ascii=False, indent=2)
            print(f"Wrote {len(parts[name])} records to {out_path}")


if __name__ == "__main__":
    main()
