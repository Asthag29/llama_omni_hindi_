
#! to build manifest from wavs files in case the job failed in the middle
import json
import os
from pathlib import Path

import yaml

FIRST_TURN_PROMPT = "<speech>\nकृपया उपयोगकर्ता के भाषण में प्रश्नों का सीधे उत्तर दें।"

with open("config.yaml", "r") as f:
    data_path = yaml.safe_load(f)["data"]["path"]

JSON_PATH = os.path.join(data_path, "combined_dataset.json")
OUT_DIR = os.path.join(data_path, "wav")
MANIFEST_PATH = os.path.join(data_path, "manifestss.json")


def build_manifest_entry(item: dict, wav_dir: str) -> list[dict]:
    item_id = item["id"]
    messages = item.get("messages", [])
    if not messages:
        return []

    first_user = next((m for m in messages if m.get("role") == "user"), None)
    first_assistant = next((m for m in messages if m.get("role") == "assistant"), None)
    if not first_user or not first_assistant:
        return []

    results = []
    wav_path = os.path.join(wav_dir, f"{item_id}-1_user.wav")

    if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
        results.append({
            "id": f"{item_id}_speech",
            "speech": wav_path,
            "conversations": [
                {"from": "human", "value": FIRST_TURN_PROMPT},
                {"from": "gpt", "value": first_assistant["content"].strip()},
            ],
        })

    num_user_turns = sum(1 for m in messages if m.get("role") == "user")
    if num_user_turns > 1:
        conversations = [
            {
                "from": "human" if m["role"] == "user" else "gpt",
                "value": m["content"].strip(),
            }
            for m in messages
        ]
        results.append({
            "id": f"{item_id}_text",
            "conversations": conversations,
        })

    return results


def main():
    os.chdir(Path(__file__).resolve().parents[3])
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    manifest = []
    missing_wav = 0
    for item in data:
        entries = build_manifest_entry(item, OUT_DIR)
        if not entries:
            missing_wav += 1
        manifest.extend(entries)

    os.makedirs(os.path.dirname(MANIFEST_PATH) or ".", exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    speech_count = sum(1 for e in manifest if "speech" in e)
    text_count = sum(1 for e in manifest if "speech" not in e)
    print(f"Wrote {len(manifest)} entries → {MANIFEST_PATH}")
    print(f"  speech entries: {speech_count}")
    print(f"  text-only multi-turn: {text_count}")
    print(f"  source items with no usable entry: {missing_wav}")


if __name__ == "__main__":
    main()