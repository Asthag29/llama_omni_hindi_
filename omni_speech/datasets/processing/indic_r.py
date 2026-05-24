import os
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

from datasets import load_from_disk
import soundfile as sf

ds = load_from_disk("data/processed/indicvoices_r_hindi/train")

print(ds.column_names)