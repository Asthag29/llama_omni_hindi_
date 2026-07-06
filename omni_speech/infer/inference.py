#!/usr/bin/env python3
"""Run OmniSpeech inference on data/inference.wav using a streaming checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisper
from omegaconf import OmegaConf

REPO_ROOT = Path("/dss/dsshome1/0C/ra85muk2/Desktop/Programming/hindi_llama_omni")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omni_speech.conversation import conv_templates
from omni_speech.datasets.preprocess import tokenizer_speech_token
from omni_speech.trainer_combined import OmniSpeechTrainingModule
from omni_speech.train_utils import (
    is_safetensors_checkpoint,
    load_omni_speech_checkpoint,
    model_dtype,
    resolve_checkpoint_path,
)

AUDIO_PATH = REPO_ROOT / "data" / "inference.wav"
CONFIG_PATH = REPO_ROOT / "configs" / "stage_2.yaml"

# Leave CHECKPOINT_PATH as None to auto-pick the latest streaming-trainer checkpoint.
# Default run written by stage-2 streaming training:
# outputs/stage_2/speech_text/checkpoints/last
CHECKPOINT_PATH = None
STREAMING_RUN_ID = "speech_text"

CONV_MODE = "llama_3"
DEFAULT_PROMPT = (
    "<speech>\n"
    "आप हिंदी लामा मॉडल हैं। उपयोगकर्ता की आवाज़ सुनें और उनके प्रश्न का उत्तर हिंदी "
    "(देवनागरी लिपि) में दें। "
    "Roman/Latin अक्षरों (Hinglish) का उपयोग न करें।"
)

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = None
NUM_BEAMS = 1

CHECKPOINT_MARKERS = (
    "adapter_model.safetensors",
    "speech_projector.safetensors",
    "trainable.safetensors",
)


def has_checkpoint_marker(path: Path) -> bool:
    return path.is_dir() and any((path / marker).is_file() for marker in CHECKPOINT_MARKERS)


def checkpoint_sort_key(path: Path):
    if path.name == "final_model":
        return (0, 0.0, -path.stat().st_mtime)

    meta_path = path / "checkpoint_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if "val_loss" in meta:
            return (1, float(meta["val_loss"]), -path.stat().st_mtime)

    return (2, 0.0, -path.stat().st_mtime)


def find_streaming_checkpoint(
    checkpoint_path: str | Path | None = CHECKPOINT_PATH,
    streaming_run_id: str | None = STREAMING_RUN_ID,
) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path).expanduser().resolve()

    candidates = []
    run_roots = []
    streaming_root = REPO_ROOT / "outputs" / "stage_2"
    configured_root = REPO_ROOT / "outputs" / "stage_2" / "speech_text"

    if streaming_run_id:
        run_roots.append(streaming_root / streaming_run_id)
    elif streaming_root.is_dir():
        run_roots.extend(sorted(path for path in streaming_root.iterdir() if path.is_dir()))
    run_roots.append(configured_root)

    seen = set()
    fallback_candidates = []
    for root in run_roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)

        preferred = [root / "checkpoints" / "last", root / "final_model"]
        for path in preferred:
            if has_checkpoint_marker(path):
                return path

        ckpt_root = root / "checkpoints"
        if ckpt_root.is_dir():
            fallback_candidates.extend(path for path in ckpt_root.iterdir() if has_checkpoint_marker(path))

    candidates.extend(fallback_candidates)

    if not candidates:
        raise FileNotFoundError(
            "No streaming safetensors checkpoint found. Set --checkpoint manually, "
            "for example outputs/stage_2/<run_id>/checkpoints/last."
        )

    return sorted(set(candidates), key=checkpoint_sort_key)[0]


def load_inference_cfg(config_path: Path):
    cfg = OmegaConf.load(config_path)
    if "hydra" in cfg:
        del cfg["hydra"]

    cfg.model.path = str((REPO_ROOT / cfg.model.path).resolve())
    cfg.model.config_path = str((REPO_ROOT / cfg.model.config_path).resolve())
    cfg.model.model_base = str((REPO_ROOT / cfg.model.model_base).resolve())
    cfg.model.tokenizer_path = str((REPO_ROOT / cfg.model.tokenizer_path).resolve())

    # The selected streaming checkpoint already contains the final LoRA adapter
    # and speech projector, so do not load the backbone init checkpoint first.
    cfg.model.init_checkpoint = None
    cfg.training.gradient_checkpointing = False
    cfg.training.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    cfg.training.devices = 1
    cfg.training.strategy = "auto"
    if not torch.cuda.is_available():
        cfg.training.precision = "32-true"
    return cfg


def load_module_from_checkpoint(checkpoint_path: Path, cfg):
    checkpoint_path = Path(resolve_checkpoint_path(str(checkpoint_path)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model on {device}...")
    module = OmniSpeechTrainingModule(cfg)

    if is_safetensors_checkpoint(str(checkpoint_path)):
        load_omni_speech_checkpoint(module, str(checkpoint_path))
    else:
        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = module.load_state_dict(checkpoint_obj["state_dict"], strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

    module.eval().to(device)
    module.model.config.use_cache = True
    return module, device


def load_audio_16k(audio_path: Path, sample_rate: int = 16000) -> np.ndarray:
    audio, file_sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if file_sr != sample_rate:
        waveform = torch.from_numpy(audio).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, int(file_sr), sample_rate)
        audio = waveform.squeeze(0).numpy()
    return audio.astype(np.float32)


def prepare_speech(audio_path: Path, cfg, module, device: torch.device):
    audio = load_audio_16k(audio_path)
    input_type = str(cfg.data.input_type)
    mel_size = int(cfg.data.mel_size)
    dtype = model_dtype(cfg.training.precision) if device.type == "cuda" else torch.float32

    if input_type == "raw":
        speech = torch.from_numpy(audio)
        inner_model = module._get_inner_speech_model()
        if getattr(inner_model.config, "speech_normalize", False):
            speech = torch.nn.functional.layer_norm(speech, speech.shape)
    elif input_type == "mel":
        if bool(cfg.data.get("compute_mel_on_gpu", False)):
            raise ValueError(
                "Script generation expects precomputed mel features; set cfg.data.compute_mel_on_gpu=False."
            )
        audio = whisper.pad_or_trim(audio)
        speech = whisper.log_mel_spectrogram(audio, n_mels=mel_size).permute(1, 0)
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")

    speech_lengths = torch.tensor([speech.shape[0]], device=device, dtype=torch.long)
    speech = speech.unsqueeze(0).to(device=device, dtype=dtype)
    return speech, speech_lengths


def build_prompt(user_text: str = DEFAULT_PROMPT, conv_mode: str = CONV_MODE) -> str:
    if "<speech>" not in user_text:
        user_text = "<speech>\n" + user_text
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


@torch.inference_mode()
def generate_from_wav(
    audio_path: Path,
    prompt: str,
    cfg,
    module,
    device: torch.device,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float | None = TOP_P,
    num_beams: int = NUM_BEAMS,
):
    model = module.model
    tokenizer = module.tokenizer
    rendered_prompt = build_prompt(prompt)
    input_ids = tokenizer_speech_token(rendered_prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
    speech, speech_lengths = prepare_speech(audio_path, cfg, module, device)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kwargs = {
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    output_ids = model.generate(
        input_ids,
        speech=speech,
        speech_lengths=speech_lengths,
        **gen_kwargs,
    )
    new_token_ids = output_ids[:, input_ids.shape[1] :] if output_ids.shape[1] > input_ids.shape[1] else output_ids
    return tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0].strip()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=AUDIO_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--streaming-run-id", default=STREAMING_RUN_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--num-beams", type=int, default=NUM_BEAMS)
    return parser.parse_args()


def main():
    args = parse_args()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    audio_path = args.audio.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    checkpoint = find_streaming_checkpoint(args.checkpoint, args.streaming_run_id)
    print(f"Repo root: {REPO_ROOT}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Selected checkpoint: {checkpoint}")

    meta_path = checkpoint / "checkpoint_meta.json"
    if meta_path.is_file():
        print(meta_path.read_text(encoding="utf-8"))

    cfg = load_inference_cfg(config_path)
    module, device = load_module_from_checkpoint(checkpoint, cfg)
    response = generate_from_wav(
        audio_path=audio_path,
        prompt=args.prompt,
        cfg=cfg,
        module=module,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
    )
    print("\n=== Model response ===")
    print(response)


if __name__ == "__main__":
    main()
