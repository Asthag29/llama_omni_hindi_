"""
Synthesize user-turn Hindi text → WAV + manifest.
Uses omni_speech.infer.hindi_tts.HindiTTSBridge (facebook/mms-tts-hin).

  # All data, one GPU
  python omni_speech/datasets/processing/text_to_audio.py

  # SLURM shard (see audio.sh)
  python omni_speech/datasets/processing/text_to_audio.py --shard-id 0 --num-shards 3

  # After all shards finish
  python omni_speech/datasets/processing/text_to_audio.py --merge --num-shards 3
"""
import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from omni_speech.infer.hindi_tts import HindiTTSBridge

JSON_PATH = "data/processed/hindi_dataset_backbone.json"
OUT_DIR = "data/processed/hindi_wav/user"
MANIFEST_PATH = "data/processed/hindi_wav/manifest_user.json"
MANIFEST_DIR = "data/processed/hindi_wav/manifest_shards"


def get_user_text(item: dict) -> str | None:
    for m in item.get("messages", []):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return None


def shard_indices(n: int, shard_id: int, num_shards: int) -> range:
    start = (n * shard_id) // num_shards
    end = (n * (shard_id + 1)) // num_shards
    return range(start, end)


def shard_manifest_path(shard_id: int, num_shards: int) -> str:
    return os.path.join(
        MANIFEST_DIR,
        f"manifest_user_shard{shard_id:02d}of{num_shards:02d}.json",
    )


def run_tts(
    data: list,
    indices: range,
    out_dir: str,
    manifest_path: str,
    device: str,
    skip_existing: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)

    tts = HindiTTSBridge(device=device)
    manifest = []
    skipped = 0

    for idx in tqdm(list(indices), desc="TTS"):
        text = get_user_text(data[idx])
        if not text:
            skipped += 1
            continue

        wav_path = os.path.join(out_dir, f"{idx:06d}.wav")
        entry = {
            "id": f"backbone_user_{idx:06d}",
            "index": idx,
            "wav": wav_path,
            "user_text": text,
        }

        if skip_existing and os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
            manifest.append(entry)
            continue

        try:
            tts.save(tts.synthesize(text), wav_path)
            manifest.append(entry)
        except Exception as e:
            print(f"[skip {idx}] {e}: {text[:80]}...")
            skipped += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(manifest)} entries → {manifest_path} (skipped {skipped})")


def merge_manifests(num_shards: int, out_path: str = MANIFEST_PATH) -> None:
    merged = []
    for shard_id in range(num_shards):
        path = shard_manifest_path(shard_id, num_shards)
        if not os.path.isfile(path):
            raise SystemExit(f"Missing shard manifest: {path}")
        with open(path, encoding="utf-8") as f:
            merged.extend(json.load(f))

    merged.sort(key=lambda x: x["index"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Merged {len(merged)} entries → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    #merging all the shard manifests into a single manifest
    p.add_argument("--merge", action="store_true", help="Combine shard manifests")
    p.add_argument("--shard-id", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--max-samples", type=int, default=None)
    #skipping the existing wav files
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    os.chdir(Path(__file__).resolve().parents[3])

    if args.merge:
        merge_manifests(args.num_shards)
        return

    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    num_shards = max(1, args.num_shards)
    if num_shards == 1:
        shard_id = 0
        manifest_path = MANIFEST_PATH
    else:
        if args.shard_id is None:
            raise SystemExit("Use --shard-id with --num-shards > 1")
        shard_id = args.shard_id
        manifest_path = shard_manifest_path(shard_id, num_shards)

    indices = shard_indices(len(data), shard_id, num_shards)
    run_tts(data, indices, OUT_DIR, manifest_path, args.device, args.skip_existing)


if __name__ == "__main__":
    main()