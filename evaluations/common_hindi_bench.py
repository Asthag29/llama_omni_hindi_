#!/usr/bin/env python3
"""Hindi-focused text evaluation for backbone checkpoints.

This compares a text-only base checkpoint vs a fine-tuned LoRA adapter on
compact Hindi benchmarks that are easy to score locally:

- `mteb/IndicSentiment` (Hindi split)
- `AdaMLLab/indicxnli_repaired` (Hindi split)
- `l3cube-pune/IndicQuest` (Hindi CSV)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Sequence

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omni_speech.conversation import conv_templates
from omni_speech.model.language_model.omni_speech_llama import (
    OmniSpeechConfig,
    OmniSpeechLlamaForCausalLM,
)

XNLI_ID_TO_LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}

SENTIMENT_CHOICES = [
    ("सकारात्मक", "Positive"),
    ("नकारात्मक", "Negative"),
    ("तटस्थ", "Neutral"),
]

XNLI_CHOICES = [
    ("अनुमिति", "entailment"),
    ("तटस्थ", "neutral"),
    ("विरोधाभास", "contradiction"),
]

INDICQUEST_HINDI_URL = "https://huggingface.co/datasets/l3cube-pune/IndicQuest/resolve/main/Hindi.csv"


def _dtype_for_device(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def _build_prompt(user_text: str, conv_mode: str = "llama_3") -> str:
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


@torch.inference_mode()
def _score_candidate(tokenizer, model, prompt: str, candidate: str, device: str) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    full_ids = tokenizer(prompt + candidate, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    if full_ids.shape[1] <= prompt_ids.shape[1]:
        return float("-inf")

    attention_mask = torch.ones_like(full_ids, device=device)
    outputs = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = outputs.logits[:, :-1, :]
    target_ids = full_ids[:, 1:]

    candidate_start = prompt_ids.shape[1] - 1
    log_probs = torch.log_softmax(logits[:, candidate_start:, :], dim=-1)
    candidate_target_ids = target_ids[:, candidate_start:]
    token_log_probs = log_probs.gather(-1, candidate_target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item())


def _predict_by_label_scoring(
    tokenizer,
    model,
    prompt: str,
    choices: Sequence[tuple[str, str]],
    device: str,
) -> tuple[str, str]:
    scored = []
    for candidate_text, canonical_label in choices:
        score = _score_candidate(tokenizer, model, prompt, candidate_text, device=device)
        scored.append((score, candidate_text, canonical_label))
    best_score, best_text, best_label = max(scored, key=lambda item: item[0])
    return best_label, best_text


def _strip_speech_config(config: OmniSpeechConfig) -> OmniSpeechConfig:
    """Remove speech-only config fields so the text backbone loads without Whisper."""
    speech_only_fields = [
        "speech_encoder",
        "speech_encoder_type",
        "speech_encoder_ds_rate",
        "speech_encoder_hidden_size",
        "speech_projector_type",
        "speech_projector_lr",
        "speech_generator_type",
        "speech_normalize",
        "ctc_decoder_config",
        "ctc_loss_weight",
        "ctc_upsample_factor",
        "unit_vocab_size",
        "freeze_speech_projector",
        "tune_speech_projector",
    ]
    for field in speech_only_fields:
        if hasattr(config, field):
            delattr(config, field)
    return config


def _load_base_model(
    model_base: str,
    config_path: str,
    tokenizer_path: str,
    device: str,
):
    config = _strip_speech_config(OmniSpeechConfig.from_pretrained(config_path))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = getattr(config, "tokenizer_model_max_length", 2048)

    model = OmniSpeechLlamaForCausalLM.from_pretrained(
        model_base,
        config=config,
        torch_dtype=_dtype_for_device(device),
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return tokenizer, model


def load_base_model(device: str):
    base = REPO_ROOT / "models" / "llama"
    return _load_base_model(str(base), str(base), str(base), device)


def _latest_lora_dir() -> Path:
    roots = sorted(
        (REPO_ROOT / "outputs" / "stage_1").glob("*/final_model"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not roots:
        raise FileNotFoundError("No fine-tuned final_model found under outputs/stage_1/*/final_model")
    return roots[0]


def load_lora_model(device: str, adapter_dir: str | None = None):
    adapter_path = Path(adapter_dir) if adapter_dir else _latest_lora_dir()
    adapter_cfg_path = adapter_path / "adapter_config.json"
    with adapter_cfg_path.open(encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_path = adapter_cfg["base_model_name_or_path"]

    tokenizer, model = _load_base_model(
        model_base=base_path,
        config_path=base_path,
        tokenizer_path=str(adapter_path),
        device=device,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    model.to(device)
    return tokenizer, model, str(adapter_path)


def _load_indic_sentiment(limit: int | None):
    ds = load_dataset("mteb/IndicSentiment", "hi", split="test")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def _load_indic_xnli(limit: int | None):
    ds = load_dataset("AdaMLLab/indicxnli_repaired", "hi", split="validation")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def _load_indicquest_hi(limit: int | None):
    # Loading the full IndicQuest repo currently fails because one CSV has a
    # mismatched header. Use the Hindi file directly for a stable evaluation.
    ds = load_dataset("csv", data_files={"train": INDICQUEST_HINDI_URL}, split="train")
    ds = ds.rename_columns({name: name.strip() for name in ds.column_names if name != name.strip()})
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def _sentiment_prompt(example: dict) -> str:
    return (
        "नीचे दी गई हिंदी समीक्षा का भाव बताइए। "
        "उत्तर केवल इन लेबलों में से एक शब्द में दें: सकारात्मक, नकारात्मक, तटस्थ.\n\n"
        f"समीक्षा: {example['INDIC REVIEW']}\n\nउत्तर:"
    )


def _xnli_prompt(example: dict) -> str:
    return (
        "नीचे दिए गए आधार-वाक्य और परिकल्पना के संबंध को पहचानिए। "
        "उत्तर केवल इन लेबलों में से एक शब्द में दें: अनुमिति, तटस्थ, विरोधाभास.\n\n"
        f"आधार-वाक्य: {example['premise']}\n"
        f"परिकल्पना: {example['hypothesis']}\n\nउत्तर:"
    )


def _indicquest_prompt(example: dict) -> str:
    return (
        "नीचे दिए गए प्रश्न का उत्तर हिंदी में दें। "
        "उत्तर संक्षिप्त और तथ्यात्मक रखें।\n\n"
        f"प्रश्न: {example['Question']}\n\nउत्तर:"
    )


@torch.inference_mode()
def _score_reference_answer(tokenizer, model, prompt: str, answer: str, device: str) -> tuple[float, float]:
    answer = " " + answer.strip()
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    full_ids = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    if full_ids.shape[1] <= prompt_ids.shape[1]:
        return float("inf"), float("inf")

    attention_mask = torch.ones_like(full_ids, device=device)
    outputs = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = outputs.logits[:, :-1, :]
    target_ids = full_ids[:, 1:]

    answer_start = prompt_ids.shape[1] - 1
    log_probs = torch.log_softmax(logits[:, answer_start:, :], dim=-1)
    answer_target_ids = target_ids[:, answer_start:]
    token_log_probs = log_probs.gather(-1, answer_target_ids.unsqueeze(-1)).squeeze(-1)

    mean_nll = -float(token_log_probs.mean().item())
    perplexity = math.exp(mean_nll) if mean_nll < 50 else float("inf")
    return mean_nll, perplexity


def _evaluate_task(
    task_name: str,
    dataset,
    prompt_fn: Callable[[dict], str],
    gold_fn: Callable[[dict], str],
    choices: Sequence[tuple[str, str]],
    tokenizer,
    model,
    device: str,
):
    correct = 0
    samples = []
    for idx, example in enumerate(dataset):
        prompt = _build_prompt(prompt_fn(example))
        pred, raw_pred = _predict_by_label_scoring(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            choices=choices,
            device=device,
        )
        gold = gold_fn(example)
        is_correct = pred == gold
        correct += int(is_correct)
        samples.append(
            {
                "index": idx,
                "gold": gold,
                "prediction": pred,
                "raw_prediction": raw_pred,
                "correct": is_correct,
            }
        )

    total = len(samples)
    return {
        "task": task_name,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
        "invalid_rate": 0.0,
        "samples": samples,
    }


def _evaluate_indicquest(
    dataset,
    tokenizer,
    model,
    device: str,
):
    samples = []
    domain_stats: dict[str, dict[str, float]] = {}
    for idx, example in enumerate(dataset):
        prompt = _build_prompt(_indicquest_prompt(example))
        gold = str(example["Answer"])
        mean_nll, perplexity = _score_reference_answer(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            answer=gold,
            device=device,
        )
        domain = str(example.get("Domain") or "unknown")

        stats = domain_stats.setdefault(domain, {"total": 0, "nll_sum": 0.0})
        stats["total"] += 1
        stats["nll_sum"] += mean_nll

        samples.append(
            {
                "index": idx,
                "domain": domain,
                "gold": gold,
                "reference_nll": mean_nll,
                "reference_perplexity": perplexity,
            }
        )
        if (idx + 1) % 25 == 0:
            print(f"indicquest_hi: evaluated {idx + 1}/{len(dataset)} samples", flush=True)

    total = len(samples)
    nll_sum = sum(float(sample["reference_nll"]) for sample in samples)
    mean_nll = (nll_sum / total) if total else 0.0
    by_domain = {
        domain: {
            "total": int(stats["total"]),
            "mean_nll": (float(stats["nll_sum"]) / int(stats["total"])) if stats["total"] else 0.0,
            "perplexity": math.exp(float(stats["nll_sum"]) / int(stats["total"])) if stats["total"] else 0.0,
        }
        for domain, stats in sorted(domain_stats.items())
    }

    return {
        "task": "indicquest_hi",
        "total": total,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if total else 0.0,
        "by_domain": by_domain,
        "samples": samples,
    }


def run_suite(args: argparse.Namespace):
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cpu":
        raise RuntimeError("A GPU device is strongly recommended for evaluating these 8B checkpoints.")

    if args.model == "base":
        tokenizer, model = load_base_model(device)
        model_ref = str(REPO_ROOT / "models" / "llama")
    else:
        tokenizer, model, model_ref = load_lora_model(device, args.checkpoint)

    sentiment = _evaluate_task(
        task_name="indic_sentiment_hi",
        dataset=_load_indic_sentiment(args.limit_sentiment),
        prompt_fn=_sentiment_prompt,
        gold_fn=lambda ex: ex["LABEL"],
        choices=SENTIMENT_CHOICES,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    xnli = _evaluate_task(
        task_name="indic_xnli_hi",
        dataset=_load_indic_xnli(args.limit_xnli),
        prompt_fn=_xnli_prompt,
        gold_fn=lambda ex: XNLI_ID_TO_LABEL[int(ex["label"])],
        choices=XNLI_CHOICES,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    indicquest = _evaluate_indicquest(
        dataset=_load_indicquest_hi(args.limit_indicquest),
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    results = {
        "model": args.model,
        "model_ref": model_ref,
        "device": device,
        "tasks": {
            sentiment["task"]: {k: v for k, v in sentiment.items() if k != "samples"},
            xnli["task"]: {k: v for k, v in xnli.items() if k != "samples"},
            indicquest["task"]: {k: v for k, v in indicquest.items() if k != "samples"},
        },
        "macro_accuracy": (sentiment["accuracy"] + xnli["accuracy"]) / 2.0,
        "samples": {
            sentiment["task"]: sentiment["samples"],
            xnli["task"]: xnli["samples"],
            indicquest["task"]: indicquest["samples"],
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "model": results["model"],
            "model_ref": results["model_ref"],
            "macro_accuracy": results["macro_accuracy"],
            "tasks": results["tasks"],
            "output": str(out_path),
        },
        ensure_ascii=False,
        indent=2,
    ))

    if args.compare_to:
        compare_path = Path(args.compare_to)
        if compare_path.is_file():
            baseline = json.loads(compare_path.read_text(encoding="utf-8"))
            print("\nComparison:")
            for task_name, task_metrics in results["tasks"].items():
                base_metrics = baseline.get("tasks", {}).get(task_name)
                if not base_metrics or "accuracy" not in task_metrics or "accuracy" not in base_metrics:
                    continue
                delta = task_metrics["accuracy"] - base_metrics["accuracy"]
                print(f"- {task_name}: {task_metrics['accuracy']:.4f} (delta {delta:+.4f})")
            base_indicquest = baseline.get("tasks", {}).get("indicquest_hi")
            curr_indicquest = results["tasks"].get("indicquest_hi")
            if base_indicquest and curr_indicquest:
                nll_delta = curr_indicquest["mean_nll"] - base_indicquest["mean_nll"]
                ppl_delta = curr_indicquest["perplexity"] - base_indicquest["perplexity"]
                print(f"- indicquest_hi mean_nll: {curr_indicquest['mean_nll']:.4f} (delta {nll_delta:+.4f})")
                print(f"- indicquest_hi perplexity: {curr_indicquest['perplexity']:.4f} (delta {ppl_delta:+.4f})")


def add_shared_args(parser: argparse.ArgumentParser, *, expose_model: bool = True) -> argparse.ArgumentParser:
    if expose_model:
        parser.add_argument("--model", choices=["base", "lora"], default="base")
    parser.add_argument("--checkpoint", default=None, help="LoRA adapter/final_model directory.")
    parser.add_argument("--output", default=None, help="Path to output JSON metrics.")
    parser.add_argument("--device", default="auto", help="Device to run on, e.g. auto or cuda.")
    parser.add_argument("--limit-sentiment", type=int, default=200)
    parser.add_argument("--limit-xnli", type=int, default=200)
    parser.add_argument("--limit-indicquest", type=int, default=200)
    parser.add_argument("--compare-to", default=None, help="Optional baseline JSON for delta reporting.")
    return parser


def main(default_model: str, default_output: Path, default_compare: Path | None = None, description: str | None = None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    add_shared_args(parser, expose_model=False)
    parser.set_defaults(
        model=default_model,
        output=str(default_output),
        compare_to=str(default_compare) if default_compare else None,
    )
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_suite(args)

