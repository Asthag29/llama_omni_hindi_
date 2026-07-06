import json
from datasets import load_dataset
from os import path
import yaml

with open('configs/combined.yaml', 'r') as f:
    config = yaml.safe_load(f)
    data_path = config['data']['path']


#! for hindi they had only 1 conversation per example
def anudesh_processing():

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
    
    output_path = path.join(data_path, "anudesh_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    return print(f"Dataset processed and saved to {output_path}")


def flan_v2_processing():
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

    output_path = path.join(data_path, "dolly_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    return print(f"Dataset processed and saved to {output_path}")

def hh_rlhf_processing():
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

    output_path = path.join(data_path, "hh_rlhf_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    print(f"Dataset processed and saved to {output_path}")

def lm_sys_processing():
    formatted = []
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
    output_path = path.join(data_path, "lm_sys_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)
    return print(f"Dataset processed and saved to {output_path}")

if __name__ == "__main__":
    anudesh_processing()