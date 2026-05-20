import json
from datasets import load_dataset
from os import path

def hindi_dataset_processing():

    dataset = load_dataset("ai4bharat/indic-instruct-data-v0.1","anudesh", split="hi")

    formatted = []
    for data in dataset:
        formatted.append({
            # "id": data["id"],
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
    
    output_path = path.join("data", "hindi_dataset_backbone.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    return print(f"Dataset processed and saved to {output_path}")

if __name__ == "__main__":
    hindi_dataset_processing()