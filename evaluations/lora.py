#!/usr/bin/env python3
from common_hindi_bench import REPO_ROOT, main


if __name__ == "__main__":
    main(
        default_model="lora",
        default_output=REPO_ROOT / "evaluations" / "results" / "lora_hindi_bench.json",
        default_compare=REPO_ROOT / "evaluations" / "results" / "base_hindi_bench.json",
        description="Evaluate the Hindi backbone LoRA.",
    )
