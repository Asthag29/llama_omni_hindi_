#!/usr/bin/env python3
"""Run IFEval-Hi through the lm-evaluation-harness fork.

Task example:
    Prompt: तीन वाक्यों में भारत के मानसून का वर्णन करें और हर वाक्य "मानसून" शब्द से शुरू करें।
    Expected behavior: The response must follow the explicit formatting/count constraints, not just answer topically.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = REPO_ROOT / "models" / "llama"
STAGE1_ADAPTER = REPO_ROOT / "outputs" / "stage_1" / "backbone_text" / "final_model"
DEFAULT_OUTPUT = REPO_ROOT / "evaluations" / "results" / "if_eval_hi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["base", "stage1"], default="stage1")
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--stage1-adapter", type=Path, default=STAGE1_ADAPTER)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", default="16", help="lm-eval batch size. Use a smaller value if generation runs out of memory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test sample limit.")
    parser.add_argument("--max-gen-toks", type=int, default=512, help="Maximum new tokens generated per IFEval-Hi prompt.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def build_model_args(args: argparse.Namespace) -> str:
    parts = [
        f"pretrained={args.base_model}",
        f"tokenizer={args.base_model}",
        f"device={args.device}",
        f"dtype={args.dtype}",
        "trust_remote_code=True",
        "low_cpu_mem_usage=False",
        "device_map=None",
    ]
    if args.model == "stage1":
        parts.append(f"peft={args.stage1_adapter}")
    return ",".join(str(part) for part in parts)


def find_lm_eval_result_json(output_dir: Path) -> Path | None:
    candidates = sorted(
        (path for path in output_dir.rglob("*.json") if "samples" not in path.name.lower()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def write_results_csv(output_dir: Path, model_name: str) -> Path | None:
    result_path = find_lm_eval_result_json(output_dir)
    if result_path is None:
        return None

    with result_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", {})
    if not isinstance(results, dict):
        return None

    csv_path = output_dir / "summary.csv"
    fieldnames = ["model", "task", "metric", "value", "source_json"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for task_name, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in metrics.items():
                if isinstance(value, int | float | str | bool) or value is None:
                    writer.writerow(
                        {
                            "model": model_name,
                            "task": task_name,
                            "metric": metric_name,
                            "value": value,
                            "source_json": str(result_path),
                        }
                    )
    return csv_path


def main() -> None:
    args = parse_args()

    # The IFEval-Hi fork imports an audio model class that is unavailable in the
    # repo-pinned Transformers version. This eval loads the full Omni model, but
    # lm-eval only supplies text input_ids, so no speech tensor is passed.
    import transformers

    if not hasattr(transformers, "Qwen2AudioForConditionalGeneration"):
        transformers.Qwen2AudioForConditionalGeneration = transformers.AutoModelForCausalLM

    # Register the full custom OmniSpeech model/config before lm-eval loads AutoModel.
    import omni_speech.model.language_model.omni_speech_llama  # noqa: F401
    from lm_eval.__main__ import cli_evaluate

    model_output_dir = args.output_path / args.model
    cli_args = [
        "lm-eval",
        "--model",
        "hf",
        "--model_args",
        build_model_args(args),
        "--tasks",
        "ifevalhi",
        "--batch_size",
        str(args.batch_size),
        "--output_path",
        str(model_output_dir),
        "--gen_kwargs",
        f"max_gen_toks={args.max_gen_toks}",
        "--log_samples",
    ]
    if args.limit is not None:
        cli_args.extend(["--limit", str(args.limit)])

    sys.argv = cli_args
    cli_evaluate()
    csv_path = write_results_csv(model_output_dir, args.model)
    if csv_path is not None:
        print(f"CSV summary written to {csv_path}")


if __name__ == "__main__":
    main()
