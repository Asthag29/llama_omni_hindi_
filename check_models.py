#!/usr/bin/env python3
"""Verify that the inference checkpoints in models/ are present and intact."""

from __future__ import annotations

import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
WHISPER_SHA256 = "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb"

REQUIRED = {
    "LLaMA-Omni": [
        MODELS_DIR / "llama" / "config.json",
        MODELS_DIR / "llama" / "tokenizer.json",
        MODELS_DIR / "llama" / "model.safetensors.index.json",
    ],
    "Whisper large-v3": [
        MODELS_DIR / "speech_encoder" / "large-v3.pt",
    ],
    "IndicF5": [
        MODELS_DIR / "indicf5" / "config.json",
        MODELS_DIR / "indicf5" / "model.py",
        MODELS_DIR / "indicf5" / "model.safetensors",
        MODELS_DIR / "indicf5" / "checkpoints" / "vocab.txt",
    ],
    "Hindi stage-2 adapter": [
        MODELS_DIR / "hindi" / "adapter_config.json",
        MODELS_DIR / "hindi" / "adapter_model.safetensors",
        MODELS_DIR / "hindi" / "speech_projector.safetensors",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    missing: list[str] = []
    for name, paths in REQUIRED.items():
        print(f"Checking {name}...")
        for path in paths:
            relative = path.relative_to(REPO_ROOT)
            if path.is_file():
                print(f"  ok  {relative}")
            else:
                print(f"  missing  {relative}")
                missing.append(str(relative))

    whisper = MODELS_DIR / "speech_encoder" / "large-v3.pt"
    if whisper.is_file():
        digest = sha256(whisper)
        if digest == WHISPER_SHA256:
            print("Whisper SHA-256 matches the official large-v3 checkpoint.")
        else:
            print("Whisper SHA-256 does not match the official large-v3 checkpoint.")
            missing.append("models/speech_encoder/large-v3.pt (checksum)")

    if missing:
        print("Checkpoint check failed:")
        for item in missing:
            print(f"  - {item}")
        return 1

    print("All inference checkpoints are present and look stable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
