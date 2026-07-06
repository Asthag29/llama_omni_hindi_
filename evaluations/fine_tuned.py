#!/usr/bin/env python3
"""Evaluate speech validation BLEU and perplexity for baseline vs streaming model."""

from __future__ import annotations

import argparse
import copy
import fnmatch
import gc
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisper
from datasets import Audio, load_dataset
from huggingface_hub import HfApi
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omni_speech.infer.inference import find_streaming_checkpoint
from omni_speech.constants import IGNORE_INDEX
from omni_speech.conversation import conv_templates
from omni_speech.datasets.preprocess import preprocess, preprocess_multimodal, tokenizer_speech_token
from omni_speech.trainer_combined import OmniSpeechTrainingModule
from omni_speech.train_utils import (
    is_safetensors_checkpoint,
    load_omni_speech_checkpoint,
    model_dtype,
    resolve_checkpoint_path,
)

FIRST_TURN_PROMPT = "<speech>\nकृपया उपयोगकर्ता के भाषण में प्रश्नों का सीधे उत्तर दें।"


@dataclass
class ValidationSample:
    sample_id: str
    conversations: list[dict]
    audio: np.ndarray
    reference: str


def resolve_repo_path(path: str | Path | None) -> str | None:
    if path in (None, ""):
        return None
    path = Path(path).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def load_eval_cfg(config_path: Path, *, load_init_checkpoint: bool, use_lora: bool):
    cfg = OmegaConf.load(config_path)
    if "hydra" in cfg:
        del cfg["hydra"]

    cfg.model.path = resolve_repo_path(cfg.model.path)
    cfg.model.config_path = resolve_repo_path(cfg.model.config_path)
    cfg.model.model_base = resolve_repo_path(cfg.model.model_base)
    cfg.model.tokenizer_path = resolve_repo_path(cfg.model.tokenizer_path)
    cfg.model.init_checkpoint = resolve_repo_path(cfg.model.get("init_checkpoint")) if load_init_checkpoint else None

    cfg.training.use_lora = bool(use_lora)
    if not use_lora:
        cfg.training.tune_llm_backbone = False
        cfg.training.tune_speech_projector = False
        cfg.training.tune_speech_encoder = False
    cfg.training.gradient_checkpointing = False
    cfg.training.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    cfg.training.devices = 1
    cfg.training.strategy = "auto"
    if not torch.cuda.is_available():
        cfg.training.precision = "32-true"
    return cfg


def list_matching_parquet_files(repo_id: str, repo_type: str, patterns: list[str]) -> list[str]:
    repo_files = HfApi().list_repo_files(repo_id=repo_id, repo_type=repo_type)
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(fnmatch.filter(repo_files, pattern))
    return sorted(set(matches))


def hf_data_url(repo_id: str, repo_type: str, path: str) -> str:
    if repo_type != "dataset":
        raise ValueError("Streaming validation expects repo_type='dataset'.")
    return f"hf://datasets/{repo_id}/{path}"


def resolve_validation_files(cfg) -> list[str]:
    streaming_cfg = cfg.streaming
    repo_id = str(streaming_cfg.repo_id)
    repo_type = str(streaming_cfg.get("repo_type", "dataset"))
    patterns = list(streaming_cfg.validation_parquet_patterns)
    matches = list_matching_parquet_files(repo_id, repo_type, patterns)
    if not matches:
        raise RuntimeError(f"No validation parquet files matched {patterns} in {repo_id}.")
    return [hf_data_url(repo_id, repo_type, path) for path in matches]


def load_streaming_dataset(data_files: list[str], cache_dir: str | None):
    dataset = load_dataset(
        "parquet",
        data_files={"validation": data_files},
        split="validation",
        streaming=True,
        cache_dir=cache_dir,
    )
    return dataset.cast_column("audio", Audio(decode=False))


def resample_if_needed(audio: np.ndarray, source_rate: int, target_rate: int = 16000) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if source_rate == target_rate:
        return audio.astype(np.float32)
    waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    waveform = torchaudio.functional.resample(waveform, source_rate, target_rate)
    return waveform.squeeze(0).numpy().astype(np.float32)


def decode_audio_value(audio_value) -> np.ndarray | None:
    if audio_value is None:
        return None
    if isinstance(audio_value, dict):
        if audio_value.get("array") is not None:
            sampling_rate = int(audio_value.get("sampling_rate") or 16000)
            return resample_if_needed(np.asarray(audio_value["array"], dtype=np.float32), sampling_rate)
        audio_bytes = audio_value.get("bytes")
        if audio_bytes is not None:
            import io

            audio, sampling_rate = sf.read(io.BytesIO(bytes(audio_bytes)), dtype="float32", always_2d=False)
            return resample_if_needed(audio, int(sampling_rate))
        return None
    if isinstance(audio_value, (bytes, bytearray, memoryview)):
        import io

        audio, sampling_rate = sf.read(io.BytesIO(bytes(audio_value)), dtype="float32", always_2d=False)
        return resample_if_needed(audio, int(sampling_rate))
    return None


def row_to_conversations(row: dict) -> list[dict]:
    conversations = row.get("conversations")
    if conversations:
        return copy.deepcopy(conversations)
    assistant_text = (row.get("assistant_text") or row.get("text") or "").strip()
    return [
        {"from": "human", "value": FIRST_TURN_PROMPT},
        {"from": "gpt", "value": assistant_text},
    ]


def collect_validation_samples(cfg, limit: int, seed: int, shuffle_buffer_size: int) -> list[ValidationSample]:
    data_files = resolve_validation_files(cfg)
    print(f"Validation parquet files: {len(data_files)}")
    dataset = load_streaming_dataset(data_files, cfg.streaming.get("cache_dir"))
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)

    samples: list[ValidationSample] = []
    skipped = 0
    for row in dataset:
        conversations = row_to_conversations(row)
        reference = ""
        if len(conversations) > 1:
            reference = str(conversations[1].get("value") or "").strip()
        if not reference:
            skipped += 1
            continue

        audio = decode_audio_value(row.get("audio"))
        if audio is None:
            skipped += 1
            continue

        samples.append(
            ValidationSample(
                sample_id=str(row.get("id") or f"validation_{len(samples):06d}"),
                conversations=conversations,
                audio=audio,
                reference=reference,
            )
        )
        if len(samples) >= limit:
            break

    print(f"Collected {len(samples)} validation samples (skipped {skipped}).")
    if len(samples) < limit:
        raise RuntimeError(f"Only collected {len(samples)} usable validation samples, requested {limit}.")
    return samples


def make_data_args(cfg):
    return type(
        "DataArgs",
        (),
        {
            "is_multimodal": True,
            "input_type": str(cfg.data.input_type),
            "mel_size": int(cfg.data.mel_size),
        },
    )()


def prepare_speech(audio: np.ndarray, cfg, model_config) -> tuple[torch.Tensor, torch.Tensor]:
    input_type = str(cfg.data.input_type)
    if input_type == "raw" or (input_type == "mel" and bool(cfg.data.get("compute_mel_on_gpu", False))):
        speech = torch.from_numpy(audio)
        if getattr(model_config, "speech_normalize", False):
            speech = torch.nn.functional.layer_norm(speech, speech.shape)
    elif input_type == "mel":
        audio = whisper.pad_or_trim(audio)
        speech = whisper.log_mel_spectrogram(audio, n_mels=int(cfg.data.mel_size)).permute(1, 0)
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")
    return speech, torch.tensor(speech.shape[0], dtype=torch.long)


def prepare_training_item(sample: ValidationSample, tokenizer, model_config, cfg):
    data_args = make_data_args(cfg)
    conversations = preprocess_multimodal([copy.deepcopy(sample.conversations)], data_args)[0]
    text = preprocess([conversations], tokenizer, has_speech=True)
    input_ids = text["input_ids"].squeeze(0)
    labels = text["labels"].squeeze(0)
    if int(input_ids.shape[-1]) > int(tokenizer.model_max_length):
        return None
    speech, speech_length = prepare_speech(sample.audio, cfg, model_config)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "speech": speech,
        "speech_length": speech_length,
    }


def collate_one(item: dict, pad_token_id: int, device: torch.device):
    input_ids = item["input_ids"].unsqueeze(0)
    labels = item["labels"].unsqueeze(0)
    attention_mask = input_ids.ne(pad_token_id)
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device),
        "speech": item["speech"].unsqueeze(0).to(device),
        "speech_lengths": item["speech_length"].unsqueeze(0).to(device),
    }


def build_generation_prompt(sample: ValidationSample, conv_mode: str) -> str:
    user_text = str(sample.conversations[0].get("value") or FIRST_TURN_PROMPT)
    if "<speech>" not in user_text:
        user_text = "<speech>\n" + user_text
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def tokenize_words(text: str) -> list[str]:
    return text.strip().split()


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def corpus_bleu(predictions: list[str], references: list[str], max_order: int = 4) -> float:
    matches = [0] * max_order
    totals = [0] * max_order
    pred_len = 0
    ref_len = 0

    for prediction, reference in zip(predictions, references):
        pred_tokens = tokenize_words(prediction)
        ref_tokens = tokenize_words(reference)
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)
        for order in range(1, max_order + 1):
            pred_ngrams = ngrams(pred_tokens, order)
            ref_ngrams = ngrams(ref_tokens, order)
            overlap = pred_ngrams & ref_ngrams
            matches[order - 1] += sum(overlap.values())
            totals[order - 1] += max(0, len(pred_tokens) - order + 1)

    if pred_len == 0:
        return 0.0

    precisions = []
    for idx, (match, total) in enumerate(zip(matches, totals)):
        if total == 0:
            precisions.append(0.0)
        elif match == 0:
            # Smooth higher-order n-grams so small Hindi samples are not always zeroed.
            precisions.append(1.0 / (2.0 * total) if idx > 0 else 0.0)
        else:
            precisions.append(match / total)

    if min(precisions) <= 0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)

    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / pred_len)
    return 100.0 * brevity_penalty * geo_mean


def load_module(name: str, cfg, checkpoint: Path | None, device: torch.device):
    print(f"\nLoading {name} on {device}...")
    module = OmniSpeechTrainingModule(cfg)
    if checkpoint is not None:
        checkpoint = Path(resolve_checkpoint_path(str(checkpoint)))
        if is_safetensors_checkpoint(str(checkpoint)):
            load_omni_speech_checkpoint(module, str(checkpoint))
        else:
            checkpoint_obj = torch.load(checkpoint, map_location="cpu")
            missing, unexpected = module.load_state_dict(checkpoint_obj["state_dict"], strict=False)
            if missing:
                print(f"{name}: missing keys while loading checkpoint: {len(missing)}")
            if unexpected:
                print(f"{name}: unexpected keys while loading checkpoint: {len(unexpected)}")
    module.eval().to(device)
    module.model.config.use_cache = True
    return module


@torch.inference_mode()
def evaluate_model(
    name: str,
    cfg,
    module,
    samples: list[ValidationSample],
    device: torch.device,
    max_new_tokens: int,
    conv_mode: str,
):
    tokenizer = module.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    speech_dtype = model_dtype(cfg.training.precision) if device.type == "cuda" else torch.float32

    predictions = []
    references = []
    records = []
    total_nll = 0.0
    total_label_tokens = 0
    skipped = 0

    for idx, sample in enumerate(samples, start=1):
        item = prepare_training_item(sample, tokenizer, module.model.config, cfg)
        if item is None:
            skipped += 1
            continue

        batch = collate_one(item, pad_token_id, device)
        labels = batch["labels"]
        label_tokens = int(labels.ne(IGNORE_INDEX).sum().item())
        if label_tokens > 0:
            outputs = module.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
                speech=batch["speech"].to(dtype=speech_dtype),
                speech_lengths=batch["speech_lengths"],
                use_cache=False,
            )
            total_nll += float(outputs.loss.item()) * label_tokens
            total_label_tokens += label_tokens

        prompt = build_generation_prompt(sample, conv_mode)
        input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
        speech = item["speech"].unsqueeze(0).to(device=device, dtype=speech_dtype)
        speech_lengths = item["speech_length"].unsqueeze(0).to(device)
        output_ids = module.model.generate(
            input_ids,
            speech=speech,
            speech_lengths=speech_lengths,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_token_ids = output_ids[:, input_ids.shape[1] :] if output_ids.shape[1] > input_ids.shape[1] else output_ids
        prediction = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0].strip()

        predictions.append(prediction)
        references.append(sample.reference)
        records.append(
            {
                "id": sample.sample_id,
                "reference": sample.reference,
                "prediction": prediction,
                "label_tokens": label_tokens,
            }
        )

        if idx == 1 or idx % 25 == 0:
            print(f"{name}: evaluated {idx}/{len(samples)} samples", flush=True)

    mean_nll = total_nll / total_label_tokens if total_label_tokens else float("nan")
    return {
        "name": name,
        "num_samples": len(records),
        "skipped": skipped,
        "bleu": corpus_bleu(predictions, references),
        "perplexity": math.exp(mean_nll) if math.isfinite(mean_nll) else float("nan"),
        "mean_nll": mean_nll,
        "total_label_tokens": total_label_tokens,
        "samples": records,
    }


def unload_module(module) -> None:
    del module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage_2.yaml")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--conv-mode", default="llama_3")
    parser.add_argument("--streaming-run-id", default="qnxitjb5")
    parser.add_argument("--streaming-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--baseline-init",
        choices=["auto", "none"],
        default="none",
        help="auto loads cfg.model.init_checkpoint; none uses only models/llama.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "evaluations" / "results" / "speech_streaming_validation.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is strongly recommended for this 8B speech evaluation.")

    baseline_cfg = load_eval_cfg(
        args.config,
        load_init_checkpoint=args.baseline_init == "auto",
        use_lora=False,
    )
    streaming_cfg = load_eval_cfg(args.config, load_init_checkpoint=False, use_lora=True)

    checkpoint = find_streaming_checkpoint(args.streaming_checkpoint, args.streaming_run_id)
    samples = collect_validation_samples(
        baseline_cfg,
        limit=args.limit,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )

    results = {
        "config": str(args.config),
        "limit": args.limit,
        "seed": args.seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "baseline_init": args.baseline_init,
        "streaming_checkpoint": str(checkpoint),
        "models": {},
    }

    baseline_module = load_module("base_no_lora", baseline_cfg, checkpoint=None, device=device)
    results["models"]["base_no_lora"] = evaluate_model(
        "base_no_lora",
        baseline_cfg,
        baseline_module,
        samples,
        device,
        max_new_tokens=args.max_new_tokens,
        conv_mode=args.conv_mode,
    )
    unload_module(baseline_module)

    streaming_module = load_module("streaming_llama", streaming_cfg, checkpoint=checkpoint, device=device)
    results["models"]["streaming_llama"] = evaluate_model(
        "streaming_llama",
        streaming_cfg,
        streaming_module,
        samples,
        device,
        max_new_tokens=args.max_new_tokens,
        conv_mode=args.conv_mode,
    )
    unload_module(streaming_module)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        name: {
            "num_samples": metrics["num_samples"],
            "bleu": metrics["bleu"],
            "perplexity": metrics["perplexity"],
            "mean_nll": metrics["mean_nll"],
        }
        for name, metrics in results["models"].items()
    }
    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
