"""Quick check for the configured Hindi combined dataset."""
import json
import os
from pathlib import Path

import yaml


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[3])
    with open("config.yaml", "r") as f:
        data_path = yaml.safe_load(f)["data"]["path"]

    candidates = [
        os.path.join(data_path, "combined_dataset.json")    ]
    json_path = next((path for path in candidates if os.path.isfile(path)), None)
    if json_path is None:
        raise FileNotFoundError("Could not find combined dataset JSON")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Dataset: {json_path}")
    print(f"Items: {len(data)}")
    if data:
        print(f"First item keys: {list(data[0].keys())}")


if __name__ == "__main__":
    main()