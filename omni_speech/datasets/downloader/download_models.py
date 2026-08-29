#!/usr/bin/env python3
"""Download the public checkpoints required for Hindi LLaMA-Omni inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlretrieve


REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"

LLAMA_OMNI = {
    "repo_id": "ICTNLP/Llama-3.1-8B-Omni",
    "revision": "4844429b4f81bc9cd93dcf5b1f0c66b8925fbab8",
    "destination": MODELS_DIR / "llama",
}
HINDI_ADAPTER = {
    "repo_id": "Pastaaaaa2003/hindi-llama-omni-model",
    "revision": "0f058bdea706c073495263297d6b767fb7044915",
    "allow_patterns": [
        "models/hindi_ckpt/stage_2/speech_text/checkpoints/last/*",
    ],
}
WHISPER_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/"
    "large-v3.pt"
)
WHISPER_SHA256 = "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_hub_model(model: dict, destination: Path | None = None) -> None:
    from huggingface_hub import snapshot_download

    destination = destination or model.get("destination") or REPO_ROOT
    print(f"Downloading {model['repo_id']}...")
    snapshot_download(
        repo_id=model["repo_id"],
        revision=model["revision"],
        local_dir=destination,
        allow_patterns=model.get("allow_patterns"),
    )


def download_whisper() -> None:
    destination = MODELS_DIR / "speech_encoder" / "large-v3.pt"
    if destination.is_file() and sha256(destination) == WHISPER_SHA256:
        print("Whisper large-v3 is already verified.")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.download")
    print("Downloading OpenAI Whisper large-v3...")
    urlretrieve(WHISPER_URL, temporary)
    if sha256(temporary) != WHISPER_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Whisper download failed its SHA-256 verification.")
    temporary.replace(destination)


def main() -> None:
    download_hub_model(LLAMA_OMNI)
    download_whisper()
    download_hub_model(HINDI_ADAPTER, REPO_ROOT)
    print("All inference checkpoints are ready in models/.")


if __name__ == "__main__":
    main()
