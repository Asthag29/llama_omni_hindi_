#!/usr/bin/env python3
"""Evaluate base and stage-1 checkpoints on NVIDIA GSM8K-Hi.

Task example:
    Question: जेनेट की बत्तखें प्रतिदिन 16 अंडे देती हैं...
    Gold final answer: 18
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omni_speech.conversation import conv_templates  # noqa: E402
from omni_speech.model.language_model.omni_speech_llama import (  # noqa: E402
    OmniSpeechConfig,
    OmniSpeechLlamaForCausalLM,
)

DATASET = "nvidia/GSM8K-Hi"
DEFAULT_OUTPUT = REPO_ROOT / "evaluations" / "results" / "gsm8k_hi.json"
STAGE1_ADAPTER = REPO_ROOT / "outputs" / "stage_1" / "backbone_text" / "final_model"


def dtype_for_device(device: str) -> torch.dtype:
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def build_prompt(user_text: str, conv_mode: str = "llama_3") -> str:
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_full_omni_model(model_base: str, config_path: str, tokenizer_path: str, device: str):
    config = OmniSpeechConfig.from_pretrained(config_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = getattr(config, "tokenizer_model_max_length", 2048)

    model = OmniSpeechLlamaForCausalLM.from_pretrained(
        model_base,
        config=config,
        torch_dtype=dtype_for_device(device),
        low_cpu_mem_usage=False,
    )
    model.eval()
    model.to(device)
    return tokenizer, model


def load_base_model(device: str):
    base = REPO_ROOT / "models" / "llama"
    return load_full_omni_model(str(base), str(base), str(base), device)


def load_lora_model(device: str, adapter_dir: str):
    adapter_path = Path(adapter_dir)
    with (adapter_path / "adapter_config.json").open(encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_path = adapter_cfg["base_model_name_or_path"]
    tokenizer, model = load_full_omni_model(base_path, base_path, str(adapter_path), device)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    model.to(device)
    return tokenizer, model, str(adapter_path)


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_dataset_rows(limit: int | None) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    page_size = 100
    dataset_param = urllib.parse.quote(DATASET, safe="")
    while True:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={dataset_param}&config=default&split=test&offset={offset}&length={page_size}"
        )
        with urllib.request.urlopen(url) as response:
            payload = json.loads(response.read().decode("utf-8"))
        page = [item["row"] for item in payload["rows"]]
        rows.extend(page)
        if limit is not None and len(rows) >= limit:
            return rows[:limit]
        if len(rows) >= payload.get("num_rows_total", len(rows)) or not page:
            return rows
        offset += page_size


def gsm8k_prompt(question: str) -> str:
    return (
        "निम्न गणित प्रश्न को चरण-दर-चरण हल करें। "
        "अंत में अंतिम उत्तर को इस प्रारूप में लिखें: #### <उत्तर>\n\n"
        f"प्रश्न: {question}\n\nसमाधान:"
    )


def prompt_token_length(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False, verbose=False)["input_ids"])


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")


@torch.inference_mode()
def generate_answers_batch(tokenizer, model, prompts: list[str], device: str, max_new_tokens: int) -> list[str]:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    finally:
        tokenizer.padding_side = original_padding_side

    input_width = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id(tokenizer),
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    generated_ids = output_ids[:, input_width:] if output_ids.shape[1] > input_width else output_ids
    return [text.strip() for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]


def normalize_number(text: str) -> str | None:
    if text is None:
        return None
    cleaned = text.replace(",", "").replace("$", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    value = match.group(0)
    if "." in value:
        return str(float(value)).rstrip("0").rstrip(".")
    return str(int(value))


def extract_gold_answer(answer: str) -> str | None:
    if "####" in answer:
        return normalize_number(answer.rsplit("####", 1)[1])
    return normalize_number(answer)


def extract_prediction_answer(prediction: str) -> str | None:
    if "####" in prediction:
        return normalize_number(prediction.rsplit("####", 1)[1])
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?", prediction)
    return normalize_number(matches[-1]) if matches else None


def evaluate_model(
    name: str,
    model_ref: str,
    tokenizer,
    model,
    dataset: list[dict],
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
    batch_size: int,
) -> dict:
    eligible = []
    skipped = {"prompt_too_long": 0, "missing_gold": 0}
    for idx, row in enumerate(dataset):
        gold_answer = extract_gold_answer(row["answer"])
        if gold_answer is None:
            skipped["missing_gold"] += 1
            continue
        prompt = build_prompt(gsm8k_prompt(row["question"]))
        prompt_tokens = prompt_token_length(tokenizer, prompt)
        if prompt_tokens > max_input_tokens:
            skipped["prompt_too_long"] += 1
            continue
        eligible.append(
            {
                "index": idx,
                "question": row["question"],
                "gold_solution": row["answer"],
                "gold_answer": gold_answer,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
            }
        )

    samples = []
    for chunk in batched(eligible, batch_size):
        predictions = generate_answers_batch(
            tokenizer,
            model,
            [item["prompt"] for item in chunk],
            device,
            max_new_tokens,
        )
        for item, prediction in zip(chunk, predictions, strict=True):
            predicted_answer = extract_prediction_answer(prediction)
            correct = predicted_answer == item["gold_answer"]
            samples.append(
                {
                    "index": item["index"],
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                    "predicted_answer": predicted_answer,
                    "prediction": prediction,
                    "correct": correct,
                    "prompt_tokens": item["prompt_tokens"],
                }
            )
        if len(samples) % 25 < len(chunk):
            print(f"{name}: evaluated {len(samples)}/{len(eligible)} GSM8K-Hi samples", flush=True)

    total = len(samples)
    correct = sum(int(sample["correct"]) for sample in samples)
    invalid = sum(int(sample["predicted_answer"] is None) for sample in samples)
    return {
        "name": name,
        "model_ref": model_ref,
        "total": total,
        "correct": correct,
        "invalid_predictions": invalid,
        "accuracy": correct / total if total else 0.0,
        "skipped": skipped,
        "samples": samples,
    }


def write_results_csv(results: dict, output_path: Path) -> Path:
    csv_path = output_path.with_suffix(".csv")
    fieldnames = ["model", "accuracy", "correct", "total", "invalid_predictions"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in results["models"].items():
            writer.writerow(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "correct": metrics["correct"],
                    "total": metrics["total"],
                    "invalid_predictions": metrics["invalid_predictions"],
                }
            )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["base", "stage1", "both"], default="both")
    parser.add_argument("--checkpoint", default=str(STAGE1_ADAPTER))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    device = resolve_device(args.device)
    if device == "cpu":
        raise RuntimeError("A GPU is required for evaluating these 8B checkpoints.")

    dataset = load_dataset_rows(args.limit)
    results = {
        "task": "gsm8k_hi",
        "dataset": {"repo": DATASET, "config": "default", "split": "test"},
        "limit": args.limit,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "device": device,
        "models": {},
    }

    if args.model in {"base", "both"}:
        tokenizer, model = load_base_model(device)
        results["models"]["base"] = evaluate_model(
            "base",
            str(REPO_ROOT / "models" / "llama"),
            tokenizer,
            model,
            dataset,
            device,
            args.max_input_tokens,
            args.max_new_tokens,
            args.batch_size,
        )
        release_model(model)

    if args.model in {"stage1", "both"}:
        tokenizer, model, model_ref = load_lora_model(device, args.checkpoint)
        results["models"]["stage1"] = evaluate_model(
            "stage1",
            model_ref,
            tokenizer,
            model,
            dataset,
            device,
            args.max_input_tokens,
            args.max_new_tokens,
            args.batch_size,
        )
        release_model(model)

    if "base" in results["models"] and "stage1" in results["models"]:
        base = results["models"]["base"]
        stage1 = results["models"]["stage1"]
        results["comparison"] = {"stage1_minus_base_accuracy": stage1["accuracy"] - base["accuracy"]}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = write_results_csv(results, out_path)

    summary = {
        "task": results["task"],
        "limit": results["limit"],
        "batch_size": results["batch_size"],
        "models": {
            name: {
                "total": metrics["total"],
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "invalid_predictions": metrics["invalid_predictions"],
                "model_ref": metrics["model_ref"],
            }
            for name, metrics in results["models"].items()
        },
        "comparison": results.get("comparison"),
        "output": str(out_path),
        "csv_output": str(csv_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
