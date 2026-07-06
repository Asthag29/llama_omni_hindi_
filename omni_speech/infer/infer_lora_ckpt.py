#!/usr/bin/env python3
"""Speech-to-text inference from a PyTorch Lightning LoRA checkpoint."""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisper
from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from omni_speech.utils import (
    is_safetensors_checkpoint,
    load_omni_speech_checkpoint,
    resolve_checkpoint_path,
)
from omni_speech.conversation import conv_templates
from omni_speech.datasets.preprocess import tokenizer_speech_token
from omni_speech.trainer import OmniSpeechTrainingModule

DEFAULT_PROMPT = (
    "<speech>\n"
    "उपयोगकर्ता की आवाज़ सुनें और प्रश्न का उत्तर केवल देवनागरी लिपि में हिंदी में दें। "
    "Roman/Latin अक्षरों (Hinglish) का उपयोग न करें।"
)


def _resolve_path(path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def _load_cfg(config_path: str) -> OmegaConf:
    cfg = OmegaConf.load(_resolve_path(config_path))
    cfg.training.gradient_checkpointing = False
    return cfg


def _model_dtype(precision: str) -> torch.dtype:
    precision = str(precision)
    if "bf16" in precision:
        return torch.bfloat16
    if "16" in precision:
        return torch.float16
    return torch.float32


def _load_audio_16k(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load wav/flac without ffmpeg (works on compute nodes)."""
    audio, file_sr = sf.read(_resolve_path(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if file_sr != sample_rate:
        waveform = torch.from_numpy(audio).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, file_sr, sample_rate)
        audio = waveform.squeeze(0).numpy()
    return audio.astype(np.float32)


def _load_audio_mel(audio_path: str, mel_size: int = 128):
    speech = _load_audio_16k(audio_path)
    speech = whisper.pad_or_trim(speech)
    mel = whisper.log_mel_spectrogram(speech, n_mels=mel_size).permute(1, 0)
    return mel, mel.shape[0]


def _build_prompt(user_text: str, conv_mode: str = "llama_3") -> str:
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


@torch.inference_mode()
def _load_module(checkpoint: str, cfg: OmegaConf, device: torch.device):
    """Load trainable weights from a safetensors dir or a Lightning .ckpt file."""
    print(f"Loading checkpoint weights: {checkpoint}")
    module = OmniSpeechTrainingModule(cfg)

    if is_safetensors_checkpoint(checkpoint):
        load_omni_speech_checkpoint(module, checkpoint)
    else:
        checkpoint_obj = torch.load(checkpoint, map_location="cpu")
        state_dict = checkpoint_obj["state_dict"]
        del checkpoint_obj
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        del state_dict
        if missing:
            print(f"Warning: missing keys when loading checkpoint: {len(missing)}")
        if unexpected:
            print(f"Warning: unexpected keys when loading checkpoint: {len(unexpected)}")

    module.eval()
    module.to(device)
    return module


@torch.inference_mode()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU required for inference. Run with srun on a GPU node.")

    cfg = _load_cfg(args.config)
    checkpoint = resolve_checkpoint_path(_resolve_path(args.checkpoint))
    audio_path = _resolve_path(args.audio)

    module = _load_module(checkpoint, cfg, device)

    model = module.model
    tokenizer = module.tokenizer
    dtype = _model_dtype(cfg.training.precision)

    prompt = _build_prompt(args.prompt, args.conv_mode)
    input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(device)

    mel, mel_len = _load_audio_mel(audio_path, mel_size=int(cfg.data.mel_size))
    speech = mel.unsqueeze(0).to(device=device, dtype=dtype)
    speech_lengths = torch.tensor([mel_len], device=device, dtype=torch.long)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    print(f"Audio: {audio_path}")
    print(f"Prompt: {args.prompt}")
    print("Generating...")

    output_ids = model.generate(
        input_ids,
        speech=speech,
        speech_lengths=speech_lengths,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else 1.0,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        pad_token_id=pad_token_id,
    )

    if output_ids.shape[1] > input_ids.shape[1]:
        new_token_ids = output_ids[:, input_ids.shape[1] :]
    else:
        new_token_ids = output_ids

    response = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0].strip()
    print("\n=== Model response ===")
    print(response)
    return response


def parse_args():
    parser = argparse.ArgumentParser(description="Run speech QA with a Lightning LoRA checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="outputs/training_lora/checkpoints",
        help="Path to safetensors checkpoint dir or legacy .ckpt file.",
    )
    parser.add_argument(
        "--config",
        default="outputs/training_lora/csv/version_0/hparams.yaml",
        help="Training config/hparams used for the checkpoint.",
    )
    parser.add_argument(
        "--audio",
        default="WhatsApp Audio 2026-06-13 at 20.31.58.wav",
        help="Input speech file (.wav / .flac).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="User message template; must include <speech> for audio conditioning.",
    )
    parser.add_argument("--conv-mode", default="llama_3")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
