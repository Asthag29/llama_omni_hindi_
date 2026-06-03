
#! non sense file ig

from datasets import load_dataset

# Hindi train split only
dataset = load_dataset("ai4bharat/indicvoices_r", "Hindi", split="train")

# Save locally
dataset.save_to_disk("./data/processed/indicvoices_r_hindi/train")

# Optional: test split
test = load_dataset("ai4bharat/indicvoices_r", "Hindi", split="test")
test.save_to_disk("./data/processed/indicvoices_r_hindi/test")