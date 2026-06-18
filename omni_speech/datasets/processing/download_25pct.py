
#! this is the main script for downloading batches

"""
Download ~25% of Hindi-speech-instruct dataset and extract FLAC audio to data/flac/.
Reads parquet in row-group chunks to avoid OOM on login nodes.
Cleans up cached parquet after extraction to save disk.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

REPO_ID = "Pastaaaaa2003/Hindi-speech-instruct"
REPO_TYPE = "dataset"
LOCAL_FLAC_DIR = Path(__file__).parent / "data" / "flac"
LOCAL_FLAC_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi()

print(f"Listing parquet files in {REPO_ID}/data ...")
all_items = list(api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, path_in_repo="data"))
parquets = sorted([f.path for f in all_items if f.path.endswith(".parquet")])

target_count = len(parquets) // 4
selected = parquets[:target_count]
print(f"Found {len(parquets)} parquet files, downloading {len(selected)} (~25%)")

total_audio = 0
for i, pq_path in enumerate(selected):
    batch_name = os.path.basename(pq_path)
    print(f"[{i+1}/{len(selected)}] {batch_name} ...", flush=True)

    local_pq = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=pq_path,
    )

    pf = pq.ParquetFile(local_pq)
    for batch in pf.iter_batches(batch_size=50):
        table = batch.to_pydict()

        if "audio" not in table:
            print(f"  Skipping {batch_name}: no audio column")
            break

        for audio_val in table["audio"]:
            if isinstance(audio_val, dict):
                audio_bytes = audio_val.get("bytes")
                audio_path = audio_val.get("path", f"unknown_{total_audio}.flac")
            elif isinstance(audio_val, bytes):
                audio_bytes = audio_val
                audio_path = f"audio_{total_audio}.flac"
            else:
                continue

            if audio_bytes is None:
                continue

            filename = os.path.basename(audio_path)
            if not filename.endswith(".flac"):
                filename += ".flac"

            dest = LOCAL_FLAC_DIR / filename
            if not dest.exists():
                with open(dest, "wb") as f:
                    f.write(audio_bytes)

            total_audio += 1

    # Free memory and remove cached parquet to save disk
    del pf
    try:
        os.remove(local_pq)
    except OSError:
        pass

    if (i + 1) % 5 == 0:
        print(f"  Extracted {total_audio} audio files so far ...")

print(f"\nDone. {total_audio} audio files saved to {LOCAL_FLAC_DIR}")
