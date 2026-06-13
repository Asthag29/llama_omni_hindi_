"""Build the single-turn speech manifest from existing FLAC files."""
import json
import os
from pathlib import Path

import yaml

FIRST_TURN_PROMPT = "<speech>\nकृपया उपयोगकर्ता के भाषण में प्रश्नों का सीधे उत्तर दें।"

with open("config.yaml", "r") as f:
    data_path = yaml.safe_load(f)["data"]["path"]

JSON_CANDIDATES = (
    os.path.join(data_path, "combined_dataset.json"),
    os.path.join(data_path, "combineddataset.json"),
)
OUT_DIR = os.path.join(data_path, "flac")
MANIFEST_PATH = os.path.join(data_path, "manifest.json")


def get_json_path() -> str:
    for path in JSON_CANDIDATES:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not find dataset JSON. Checked: " + ", ".join(JSON_CANDIDATES)
    )


def build_manifest_entry(item: dict, audio_dir: str) -> list[dict]:
    item_id = item["id"]
    messages = item.get("messages", [])
    if len(messages) != 2:
        return []

    first_user, first_assistant = messages
    if first_user.get("role") != "user" or first_assistant.get("role") != "assistant":
        return []

    results = []
    audio_path = os.path.join(audio_dir, f"{item_id}-1_user.flac")

    if os.path.isfile(audio_path) and os.path.getsize(audio_path) > 0:
        results.append({
            "id": f"{item_id}_speech",
            "speech": audio_path,
            "conversations": [
                {"from": "human", "value": FIRST_TURN_PROMPT},
                {"from": "gpt", "value": first_assistant["content"].strip()},
            ],
        })

    return results


def main():
    os.chdir(Path(__file__).resolve().parents[3])
    json_path = get_json_path()
    print(f"Using dataset JSON: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    manifest = []
    missing_audio = 0
    for item in data:
        entries = build_manifest_entry(item, OUT_DIR)
        if not entries:
            missing_audio += 1
        manifest.extend(entries)

    os.makedirs(os.path.dirname(MANIFEST_PATH) or ".", exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    speech_count = sum(1 for e in manifest if "speech" in e)
    text_count = sum(1 for e in manifest if "speech" not in e)
    print(f"Wrote {len(manifest)} entries → {MANIFEST_PATH}")
    print(f"  speech entries: {speech_count}")
    print(f"  text-only multi-turn: {text_count}")
    print(f"  source items with no usable entry: {missing_audio}")


if __name__ == "__main__":
    main()