"""Download Hindi splits of ai4bharat/indic-instruct-data-v0.1 and save them as JSON.

Each split is written to ``<output-dir>/<split>_dataset.json`` in the
``{"id": ..., "messages": [{"role": ..., "content": ...}, ...]}`` format.
"""

import argparse
import json
import os
from os import path

from datasets import load_dataset


def _save(formatted, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)
    return output_path


#! for hindi they had only 1 conversation per example
def anudesh_processing(output_dir="data"):

    dataset = load_dataset("ai4bharat/indic-instruct-data-v0.1","anudesh", split="hi")

    formatted = []
    for data in dataset:
        formatted.append({
            "id": data["id"],
            "messages": [
                {
                    "role": data["messages"][0]["role"],
                    "content": data["messages"][0]["content"]
                },
                {
                    "role": data["messages"][1]["role"],
                    "content": data["messages"][1]["content"]
                }
            ]
        })

    return _save(formatted, output_dir, "anudesh_dataset.json")


def flan_v2_processing(output_dir="data"):
    dataset = load_dataset("ai4bharat/indic-instruct-data-v0.1", "flan_v2", split="hi")
    formatted = []
    for data in dataset:
        formatted.append({
            "id": data["id"],
            "messages": [
                {
                    "role": "user",
                    "content": data["inputs"]
                },
                {
                    "role": "assistant",
                    "content": data["targets"]
                }
            ]
        })

    return _save(formatted, output_dir, "flan_v2_dataset.json")


def hh_rlhf_processing(output_dir="data"):
    dataset = load_dataset("ai4bharat/indic-instruct-data-v0.1", "hh-rlhf", split="hi")
    formatted = []
    for data in dataset:
        conversation = []
        for msg in data["messages"]:
            conversation.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        formatted.append({
            "id": data["id"],
            "messages": conversation
        })

    return _save(formatted, output_dir, "hh_rlhf_dataset.json")


def lm_sys_processing(output_dir="data"):
    dataset = load_dataset("ai4bharat/indic-instruct-data-v0.1", "lm_sys", split="hi")
    formatted = []
    for data in dataset:
        conversations = []
        for msg in data["messages"]:
            conversations.append({
                "role": msg["role"],
                "content": msg["content"],
            })
        formatted.append({
            "id": data["id"],
            "messages": conversations,
        })

    return _save(formatted, output_dir, "lm_sys_dataset.json")


SPLITS = {
    "anudesh": anudesh_processing,
    "flan_v2": flan_v2_processing,
    "hh_rlhf": hh_rlhf_processing,
    "lm_sys": lm_sys_processing,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(SPLITS),
        default=list(SPLITS),
        help="Splits to download (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write <split>_dataset.json files into (default: data).",
    )
    args = parser.parse_args()

    for split in args.splits:
        output_path = SPLITS[split](args.output_dir)
        print(f"Dataset processed and saved to {output_path}")


if __name__ == "__main__":
    main()
