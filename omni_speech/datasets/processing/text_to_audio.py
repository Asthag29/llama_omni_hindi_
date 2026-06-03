"""
Synthesize user-turn Hindi text → WAV + training manifest.

  # Single GPU
  python omni_speech/datasets/processing/text_to_audio.py

  # Multi-GPU shards (e.g. 4 GPUs)
  python omni_speech/datasets/processing/text_to_audio.py --shard-id 0 --num-shards 4
  python omni_speech/datasets/processing/text_to_audio.py --shard-id 1 --num-shards 4
  ...

  # Merge after all shards complete
  python omni_speech/datasets/processing/text_to_audio.py --merge --num-shards 4
"""
import argparse
import json
import os
from pathlib import Path

import yaml
from tqdm import tqdm

from omni_speech.infer.hindi_tts import HindiTTSBridge

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    data_path = config["data"]["path"]

JSON_PATH = os.path.join(data_path, "combined_dataset.json")
OUT_DIR = os.path.join(data_path, "wavss")
MANIFEST_PATH = os.path.join(data_path, "manifestss.json")
MANIFEST_DIR = os.path.join(data_path, "manifestss_shardss")

FIRST_TURN_PROMPT = "<speech>\nकृपया उपयोगकर्ता के भाषण में प्रश्नों का सीधे उत्तर दें।"


def build_conversations(item: dict, wav_dir: str, tts: HindiTTSBridge, skip_existing: bool) -> list[dict]:
    """Returns 1 entry for single-turn, 2 entries for multi-turn."""
    item_id = item["id"]
    messages = item.get("messages", [])
    if not messages:
        return []

    results = []

    # --- Entry 1: Speech (always first user turn only) ---
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    first_assistant = next((m for m in messages if m.get("role") == "assistant"), None)
    if not first_user or not first_assistant:
        return []

    content = first_user["content"].strip()
    wav_name = f"{item_id}-1_user.wav"
    wav_path = os.path.join(wav_dir, wav_name)

    if not (skip_existing and os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0):
        tts.save(tts.synthesize(content), wav_path)

    results.append({
        "id": f"{item_id}_speech",
        "speech": wav_path,
        "conversations": [
            {"from": "human", "value": FIRST_TURN_PROMPT},
            {"from": "gpt", "value": first_assistant["content"].strip()},
        ],
    })

    # --- Entry 2: Text-only multi-turn (only if >1 turn) ---
    num_user_turns = sum(1 for m in messages if m.get("role") == "user")
    if num_user_turns > 1:
        conversations = []
        for msg in messages:
            conversations.append({
                "from": "human" if msg["role"] == "user" else "gpt",
                "value": msg["content"].strip(),
            })
        results.append({
            "id": f"{item_id}_text",
            "conversations": conversations,
        })

    return results

def shard_indices(n: int, shard_id: int, num_shards: int) -> range:
    start = (n * shard_id) // num_shards
    end = (n * (shard_id + 1)) // num_shards
    return range(start, end)


def shard_manifest_path(shard_id: int, num_shards: int) -> str:
    return os.path.join(MANIFEST_DIR, f"manifest_shard{shard_id:02d}of{num_shards:02d}.json")


def run_tts(data: list, indices: range, out_dir: str, manifest_path: str, device: str, skip_existing: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)

    tts = HindiTTSBridge(device=device)
    manifest = []
    skipped = 0

    for idx in tqdm(list(indices), desc="TTS"):
        try:
            entry = build_conversations(data[idx], out_dir, tts, skip_existing)
            if entry:
                manifest.extend(entry)
            else:
                skipped += 1
        except Exception as e:
            print(f"[skip {idx}] {e}")
            skipped += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(manifest)} entries → {manifest_path} (skipped {skipped})")


def merge_manifests(num_shards: int) -> None:
    merged = []
    for shard_id in range(num_shards):
        path = shard_manifest_path(shard_id, num_shards)
        if not os.path.isfile(path):
            raise SystemExit(f"Missing shard: {path}")
        with open(path, encoding="utf-8") as f:
            merged.extend(json.load(f))

    os.makedirs(os.path.dirname(MANIFEST_PATH) or ".", exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Merged {len(merged)} entries → {MANIFEST_PATH}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merge", action="store_true")
    p.add_argument("--shard-id", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    os.chdir(Path(__file__).resolve().parents[3])

    if args.merge:
        merge_manifests(args.num_shards)
        return

    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if args.max_samples:
        data = data[: args.max_samples]

    num_shards = max(1, args.num_shards)
    if num_shards == 1:
        indices = shard_indices(len(data), 0, 1)
        manifest_path = MANIFEST_PATH
    else:
        if args.shard_id is None:
            raise SystemExit("Use --shard-id with --num-shards > 1")
        indices = shard_indices(len(data), args.shard_id, num_shards)
        manifest_path = shard_manifest_path(args.shard_id, num_shards)

    run_tts(data, indices, OUT_DIR, manifest_path, args.device, args.skip_existing)


if __name__ == "__main__":
    main()