#!/usr/bin/env python3
"""Evaluate base and stage-1 checkpoints on the Hindi IndicQA test set.

Task example:
    Context: नाना साहब पेशवा बाजीराव द्वितीय के दत्तक पुत्र थे...
    Question: नाना साहब का जन्म किस वर्ष हुआ था?
    Gold answer: 1824
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import unicodedata
import urllib.request
from collections import Counter
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

INDICQA_HI_URL = "https://huggingface.co/datasets/ai4bharat/IndicQA/resolve/main/data/indicqa.hi.json"
HINDI_EQUIVALENCE_TRANSLATION = str.maketrans(
    {
        "\u093c": "",  # nukta: क़/क, फ़/फ, ज़/ज are treated as equivalent.
        "\u0901": "\u0902",  # chandrabindu/anusvara variants.
        "\u200c": "",
        "\u200d": "",
    }
)


def dtype_for_device(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def build_prompt(user_text: str, conv_mode: str = "llama_3") -> str:
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_full_omni_model(
    model_base: str,
    config_path: str,
    tokenizer_path: str,
    device: str,
):
    config = OmniSpeechConfig.from_pretrained(config_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = getattr(config, "tokenizer_model_max_length", 2048)

    # Keep the speech fields in config so the full Omni model is constructed.
    # Text-only evals pass no `speech`, so the model skips Whisper/projector at runtime.
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

    tokenizer, model = load_full_omni_model(
        model_base=base_path,
        config_path=base_path,
        tokenizer_path=str(adapter_path),
        device=device,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    model.to(device)
    return tokenizer, model, str(adapter_path)


def load_indicqa_hi(limit: int | None) -> list[dict]:
    """Load Hindi IndicQA directly because the HF repo still uses a dataset script."""
    with urllib.request.urlopen(INDICQA_HI_URL) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows = []
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                rows.append(
                    {
                        "id": qa["id"],
                        "context": context,
                        "question": qa["question"].strip(),
                        "answers": {
                            "text": [answer["text"].strip() for answer in qa["answers"]],
                            "answer_start": [answer["answer_start"] for answer in qa["answers"]],
                        },
                    }
                )

    return rows[: min(limit, len(rows))] if limit is not None else rows


def indicqa_prompt(example: dict) -> str:
    return (
        "नीचे दिए गए संदर्भ के आधार पर प्रश्न का उत्तर हिंदी में दें। "
        "उत्तर संक्षिप्त और संदर्भ के अनुसार रखें।\n\n"
        f"संदर्भ: {example['context']}\n\n"
        f"प्रश्न: {example['question']}\n\nउत्तर:"
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


def pad_sequences(sequences: list[list[int]], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
    for row_idx, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row_idx, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row_idx, :length] = 1
    return input_ids, attention_mask


@torch.inference_mode()
def score_reference_answers_batch(
    tokenizer,
    model,
    prompts: list[str],
    answers_per_prompt: list[list[str]],
    device: str,
    batch_size: int,
) -> list[tuple[float, float, str]]:
    candidates = []
    for sample_idx, (prompt, answers) in enumerate(zip(prompts, answers_per_prompt, strict=True)):
        prompt_ids = tokenizer(prompt, add_special_tokens=False, verbose=False)["input_ids"]
        for answer in answers:
            answer_text = " " + answer.strip()
            full_ids = tokenizer(prompt + answer_text, add_special_tokens=False, verbose=False)["input_ids"]
            answer_start = len(prompt_ids) - 1
            target_length = len(full_ids) - 1
            candidates.append(
                {
                    "sample_idx": sample_idx,
                    "answer": answer,
                    "full_ids": full_ids,
                    "answer_start": answer_start,
                    "target_length": target_length,
                }
            )

    best_by_sample: list[tuple[float, float, str] | None] = [None for _ in prompts]
    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate["target_length"] > candidate["answer_start"] >= 0
    ]
    for candidate_batch in batched(valid_candidates, batch_size):
        input_ids, attention_mask = pad_sequences(
            [candidate["full_ids"] for candidate in candidate_batch],
            pad_token_id(tokenizer),
            device,
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        target_mask = torch.zeros_like(target_ids, dtype=torch.bool, device=device)
        for row_idx, candidate in enumerate(candidate_batch):
            target_mask[row_idx, candidate["answer_start"] : candidate["target_length"]] = True

        nlls = -(token_log_probs * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp_min(1)
        for candidate, nll in zip(candidate_batch, nlls.tolist(), strict=True):
            mean_nll = float(nll)
            perplexity = math.exp(mean_nll) if mean_nll < 50 else float("inf")
            sample_idx = int(candidate["sample_idx"])
            current = best_by_sample[sample_idx]
            if current is None or mean_nll < current[0]:
                best_by_sample[sample_idx] = (mean_nll, perplexity, str(candidate["answer"]))

    return [
        best if best is not None else (float("inf"), float("inf"), answers[0])
        for best, answers in zip(best_by_sample, answers_per_prompt, strict=True)
    ]


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().strip())
    text = text.translate(HINDI_EQUIVALENCE_TRANSLATION)
    chars = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] in {"P", "S"}:
            chars.append(" ")
        else:
            chars.append(char)
    return " ".join("".join(chars).split())


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_contains_match(prediction: str, gold: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    if not normalized_prediction or not normalized_gold:
        return float(normalized_prediction == normalized_gold)
    return float(normalized_gold in normalized_prediction or normalized_prediction in normalized_gold)


def best_contains_f1(prediction: str, answers: list[str]) -> tuple[float, float]:
    contains_match = max(answer_contains_match(prediction, answer) for answer in answers)
    f1 = max(token_f1(prediction, answer) for answer in answers)
    return contains_match, f1


def compute_best_bertscores(
    predictions: list[str],
    references_per_sample: list[list[str]],
    model_type: str,
    device: str,
    batch_size: int,
) -> list[dict[str, float]]:
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError("BERTScore requires the `bert-score` package. Install project dependencies again.") from exc

    expanded_predictions = []
    expanded_references = []
    sample_indices = []
    for sample_idx, (prediction, references) in enumerate(zip(predictions, references_per_sample, strict=True)):
        for reference in references:
            reference = reference.strip()
            if not reference:
                continue
            expanded_predictions.append(prediction.strip() or " ")
            expanded_references.append(reference)
            sample_indices.append(sample_idx)

    scores = [{"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0} for _ in predictions]
    if not expanded_predictions:
        return scores

    precision, recall, f1 = bert_score(
        expanded_predictions,
        expanded_references,
        model_type=model_type,
        device=device,
        batch_size=batch_size,
        verbose=False,
    )
    for sample_idx, precision_value, recall_value, f1_value in zip(
        sample_indices,
        precision.tolist(),
        recall.tolist(),
        f1.tolist(),
        strict=True,
    ):
        if f1_value > scores[sample_idx]["bertscore_f1"]:
            scores[sample_idx] = {
                "bertscore_precision": float(precision_value),
                "bertscore_recall": float(recall_value),
                "bertscore_f1": float(f1_value),
            }
    return scores


def add_bertscore_to_result(result: dict, model_type: str, device: str, batch_size: int) -> None:
    samples = result["samples"]
    scores = compute_best_bertscores(
        [sample["prediction"] for sample in samples],
        [sample["all_gold_answers"] for sample in samples],
        model_type,
        device,
        batch_size,
    )
    for sample, score_values in zip(samples, scores, strict=True):
        sample.update(score_values)

    total = len(samples)
    result["bertscore_precision"] = (
        sum(float(sample["bertscore_precision"]) for sample in samples) / total if total else 0.0
    )
    result["bertscore_recall"] = (
        sum(float(sample["bertscore_recall"]) for sample in samples) / total if total else 0.0
    )
    result["bertscore_f1"] = sum(float(sample["bertscore_f1"]) for sample in samples) / total if total else 0.0


@torch.inference_mode()
def generate_answers_batch(tokenizer, model, prompts: list[str], device: str, max_new_tokens: int) -> list[str]:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
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
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


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
    samples = []
    skipped = {
        "empty_answers": 0,
        "prompt_too_long": 0,
    }
    eligible = []
    for idx, example in enumerate(dataset):
        prompt = build_prompt(indicqa_prompt(example))
        answers = [str(answer) for answer in example["answers"]["text"] if str(answer).strip()]
        if not answers:
            skipped["empty_answers"] += 1
            continue

        input_tokens = prompt_token_length(tokenizer, prompt)
        if input_tokens > max_input_tokens:
            skipped["prompt_too_long"] += 1
            continue

        eligible.append(
            {
                "index": idx,
                "id": example["id"],
                "prompt": prompt,
                "answers": answers,
                "prompt_tokens": input_tokens,
            }
        )

    for chunk in batched(eligible, batch_size):
        prompts = [item["prompt"] for item in chunk]
        answers_per_prompt = [item["answers"] for item in chunk]
        scored_answers = score_reference_answers_batch(
            tokenizer,
            model,
            prompts,
            answers_per_prompt,
            device,
            batch_size,
        )
        predictions = [
            prediction.strip()
            for prediction in generate_answers_batch(tokenizer, model, prompts, device, max_new_tokens)
        ]
        for item, prediction, scored_answer in zip(chunk, predictions, scored_answers, strict=True):
            mean_nll, perplexity, best_answer = scored_answer
            contains_match, f1 = best_contains_f1(prediction, item["answers"])
            samples.append(
                {
                    "index": item["index"],
                    "id": item["id"],
                    "gold": best_answer,
                    "all_gold_answers": item["answers"],
                    "prediction": prediction,
                    "contains_match": contains_match,
                    "f1": f1,
                    "prompt_tokens": item["prompt_tokens"],
                    "reference_nll": mean_nll,
                    "reference_perplexity": perplexity,
                }
            )
        if len(samples) % 25 < len(chunk):
            print(f"{name}: evaluated {len(samples)}/{len(eligible)} IndicQA samples", flush=True)

    total = len(samples)
    mean_nll = (sum(float(sample["reference_nll"]) for sample in samples) / total) if total else 0.0
    contains_match = (sum(float(sample["contains_match"]) for sample in samples) / total) if total else 0.0
    f1 = (sum(float(sample["f1"]) for sample in samples) / total) if total else 0.0
    return {
        "name": name,
        "model_ref": model_ref,
        "total": total,
        "skipped": skipped,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if total else 0.0,
        "contains_match": contains_match,
        "f1": f1,
        "samples": samples,
    }


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_results_csv(results: dict, output_path: Path) -> Path:
    csv_path = output_path.with_suffix(".csv")
    fieldnames = [
        "model",
        "mean_nll",
        "perplexity",
        "contains_match",
        "f1",
        "bertscore_f1",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in results["models"].items():
            writer.writerow(
                {
                    "model": model_name,
                    "mean_nll": metrics["mean_nll"],
                    "perplexity": metrics["perplexity"],
                    "contains_match": metrics["contains_match"],
                    "f1": metrics["f1"],
                    "bertscore_f1": metrics.get("bertscore_f1"),
                }
            )
    return csv_path


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["base", "stage1", "both"], default="both")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "outputs" / "stage_1" / "backbone_text" / "final_model"))
    parser.add_argument("--limit", type=int, default=None, help="Optional number of examples to evaluate. Defaults to all.")
    parser.add_argument("--max-input-tokens", type=int, default=2048, help="Skip rows where context+question prompt exceeds this token length.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum generated answer tokens for EM/F1 scoring.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for model forward/generation calls.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=str(REPO_ROOT / "evaluations" / "results" / "indicqa.json"))
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore calculation.")
    parser.add_argument("--bertscore-model", default="bert-base-multilingual-cased")
    parser.add_argument("--bertscore-device", default="cpu")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    device = resolve_device(args.device)
    if device == "cpu":
        raise RuntimeError("A GPU is required for evaluating these 8B checkpoints.")

    dataset = load_indicqa_hi(args.limit)
    results = {
        "task": "indicqa_hi",
        "dataset": {
            "repo": "ai4bharat/IndicQA",
            "config": "indicqa.hi",
            "split": "test",
            "source_url": INDICQA_HI_URL,
        },
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
        if not args.skip_bertscore:
            add_bertscore_to_result(
                results["models"]["base"],
                args.bertscore_model,
                args.bertscore_device,
                args.bertscore_batch_size,
            )

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
        if not args.skip_bertscore:
            add_bertscore_to_result(
                results["models"]["stage1"],
                args.bertscore_model,
                args.bertscore_device,
                args.bertscore_batch_size,
            )

    if "base" in results["models"] and "stage1" in results["models"]:
        base = results["models"]["base"]
        stage1 = results["models"]["stage1"]
        results["comparison"] = {
            "stage1_minus_base_mean_nll": stage1["mean_nll"] - base["mean_nll"],
            "stage1_minus_base_perplexity": stage1["perplexity"] - base["perplexity"],
            "stage1_minus_base_contains_match": stage1["contains_match"] - base["contains_match"],
            "stage1_minus_base_f1": stage1["f1"] - base["f1"],
            "stage1_minus_base_bertscore_f1": stage1.get("bertscore_f1", 0.0) - base.get("bertscore_f1", 0.0),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = write_results_csv(results, out_path)

    summary = {
        "task": results["task"],
        "limit": results["limit"],
        "max_input_tokens": results["max_input_tokens"],
        "max_new_tokens": results["max_new_tokens"],
        "batch_size": results["batch_size"],
        "models": {
            name: {
                "total": metrics["total"],
                "skipped": metrics["skipped"],
                "mean_nll": metrics["mean_nll"],
                "perplexity": metrics["perplexity"],
                "contains_match": metrics["contains_match"],
                "f1": metrics["f1"],
                "bertscore_f1": metrics.get("bertscore_f1"),
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
