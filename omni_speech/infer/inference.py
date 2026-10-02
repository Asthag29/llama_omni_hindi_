#!/usr/bin/env python3
"""Run OmniSpeech inference on data/inference.wav using a stage-2 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import whisper
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omni_speech.conversation import conv_templates
from omni_speech.constants import DEFAULT_SPEECH_PROMPT
from omni_speech.datasets.preprocess import tokenizer_speech_token
from omni_speech.training.combined import OmniSpeechTrainingModule
from omni_speech.train_utils import (
    BEST_MODEL_DIRNAME,
    FINAL_MODEL_DIRNAME,
    is_safetensors_checkpoint,
    load_audio_16k,
    load_omni_speech_checkpoint,
    model_dtype,
    resolve_checkpoint_path,
    stable_best_checkpoint_path,
)

AUDIO_PATH = REPO_ROOT / "data" / "inference.wav"
CONFIG_PATH = REPO_ROOT / "configs" / "stage_2.yaml"

# Leave CHECKPOINT_PATH as None to auto-pick models/hindi, then a stage-2 run under outputs/stage_2
# (see find_stage2_checkpoint for the order).
CHECKPOINT_PATH = None
STAGE2_RUN_ID = "speech_text"

CONV_MODE = "llama_3"
DEFAULT_PROMPT = DEFAULT_SPEECH_PROMPT

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = None
NUM_BEAMS = 1

def checkpoint_sort_key(path: Path):
    """Fallback order for other checkpoint directories: lowest val_loss first."""
    meta_path = path / "checkpoint_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if "val_loss" in meta:
            return (0, float(meta["val_loss"]), -path.stat().st_mtime)

    return (1, 0.0, -path.stat().st_mtime)


def find_stage2_checkpoint(
    checkpoint_path: str | Path | None = CHECKPOINT_PATH,
    run_id: str | None = STAGE2_RUN_ID,
) -> Path:
    """Pick the checkpoint to load: explicit path, then models/hindi, then a stage-2 run.

    Within a stage-2 run directory: best_model/, checkpoints/best (best-so-far of a
    running or killed fit), final_model/, checkpoints/last, then any other
    checkpoints/* directory with the lowest val_loss.
    """
    if checkpoint_path is not None:
        return Path(checkpoint_path).expanduser().resolve()

    published = REPO_ROOT / "models" / "hindi"
    if is_safetensors_checkpoint(published):
        return published

    stage2_root = REPO_ROOT / "outputs" / "stage_2"
    if run_id:
        run_roots = [stage2_root / run_id]
    elif stage2_root.is_dir():
        run_roots = sorted(path for path in stage2_root.iterdir() if path.is_dir())
    else:
        run_roots = []
    # Always fall back to the default stage-2 run directory.
    run_roots.append(stage2_root / STAGE2_RUN_ID)

    seen = set()
    candidates = []
    for root in run_roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)

        # best_model (completed run), then the best-so-far weights of a running or
        # killed run, then the last weights.
        preferred = [
            root / BEST_MODEL_DIRNAME,
            Path(stable_best_checkpoint_path(str(root))),
            root / FINAL_MODEL_DIRNAME,
            root / "checkpoints" / "last",
        ]
        for path in preferred:
            if is_safetensors_checkpoint(path):
                # Pin a symlink (checkpoints/best, checkpoints/last) to the directory it
                # points at now, so a running fit swapping the link cannot change it.
                return path.resolve() if path.is_symlink() else path

        ckpt_root = root / "checkpoints"
        if ckpt_root.is_dir():
            candidates.extend(path for path in ckpt_root.iterdir() if is_safetensors_checkpoint(path))

    if not candidates:
        raise FileNotFoundError(
            "No stage-2 safetensors checkpoint found. Set --checkpoint manually, "
            "for example models/hindi."
        )

    return sorted(set(candidates), key=checkpoint_sort_key)[0]


def load_inference_cfg(config_path: Path):
    cfg = OmegaConf.load(config_path)
    if "hydra" in cfg:
        del cfg["hydra"]

    cfg.model.config_path = str((REPO_ROOT / cfg.model.config_path).resolve())
    cfg.model.model_base = str((REPO_ROOT / cfg.model.model_base).resolve())
    cfg.model.tokenizer_path = str((REPO_ROOT / cfg.model.tokenizer_path).resolve())

    # The selected stage-2 checkpoint already contains the final LoRA adapter
    # and speech projector, so do not load the backbone init checkpoint first.
    cfg.model.init_checkpoint = None
    cfg.training.gradient_checkpointing = False
    cfg.training.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    cfg.training.devices = 1
    cfg.training.strategy = "auto"
    if not torch.cuda.is_available():
        cfg.training.precision = "32-true"
    return cfg


def load_module_from_checkpoint(
    checkpoint_path: Path,
    cfg,
    requested_device: str | torch.device | None = None,
):
    checkpoint_path = Path(resolve_checkpoint_path(str(checkpoint_path)))
    if requested_device is None or str(requested_device) == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)

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

    # Training keeps the trainable LoRA/projector weights in fp32 while the base is
    # half precision; generation runs without autocast, so cast everything to one
    # inference dtype (bf16 on GPU by default, fp32 for the CPU "32-true" config).
    # Only parameters are cast; buffers such as the rotary inv_freq keep their dtype.
    inference_dtype = model_dtype(cfg.training.precision)
    for param in module.parameters():
        if param.is_floating_point():
            param.data = param.data.to(inference_dtype)
    module.eval().to(device)
    module.model.config.use_cache = True
    return module, device


def speech_input_dtype(model) -> torch.dtype:
    """Dtype the speech features must have: the dtype of the loaded model's weights."""
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def prepare_speech(audio_path: Path, cfg, module, device: torch.device):
    audio = load_audio_16k(audio_path)
    input_type = str(cfg.data.input_type)
    mel_size = int(cfg.data.mel_size)
    dtype = speech_input_dtype(module.model)

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
    parser.add_argument("--run-id", "--streaming-run-id", dest="run_id", default=STAGE2_RUN_ID,
                        help="Run directory under outputs/stage_2 to search when --checkpoint is not set.")
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

    checkpoint = find_stage2_checkpoint(args.checkpoint, args.run_id)
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
