"""Convert Hindi instruct data from messages format to conversations format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json_array_maybe_prefixed(path: Path):
    text = path.read_text(encoding="utf-8")
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not locate a JSON array in {path}")
    return json.loads(text[start : end + 1])


def normalize_entry(item: dict) -> dict | None:
    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    normalized = []
    for message in messages:
        role = {"user": "human", "assistant": "gpt"}.get(message.get("role"))
        value = str(message.get("content", "")).strip()
        if role not in {"human", "gpt"} or not value:
            return None
        normalized.append({"from": role, "value": value})

    if not normalized or normalized[0]["from"] != "human":
        return None

    assistant_idx = next(
        (idx for idx, turn in enumerate(normalized[1:], start=1) if turn["from"] == "gpt"),
        None,
    )
    if assistant_idx is None:
        return None

    return {
        "id": item.get("id"),
        "conversations": [normalized[0], normalized[assistant_idx]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/instruct/hindi_instruct.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/instruct/hindi_instruct_conversations.json"),
    )
    args = parser.parse_args()

    samples = load_json_array_maybe_prefixed(args.input.resolve())

    converted = []
    skipped = 0
    for item in samples:
        normalized = normalize_entry(item)
        if normalized is None:
            skipped += 1
            continue
        converted.append(normalized)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(
        f"Converted {len(converted)} samples to {args.output} "
        f"(skipped {skipped} malformed entries)."
    )


if __name__ == "__main__":
    main()
