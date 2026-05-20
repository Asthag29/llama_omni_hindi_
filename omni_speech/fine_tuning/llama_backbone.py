from omni_speech.fine_tuning.lora_config import lora_config
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
import json, torch

model, tokenizer = lora_config()

def train_llama_backbone():
    training_args = TrainingArguments(
        output_dir='./checkpoints/hindi_lora',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=2e-4,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        bf16=True, fp16=False,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        evaluation_strategy='steps',
        eval_steps=200,
        dataloader_num_workers=4,
        report_to='tensorboard',
        gradient_checkpointing=True, # effective batch = 16
        optim='adamw_torch',
        max_grad_norm=1.0,
        )
    # Load your prepared data
    with open('data/hindi_dataset_backbone.json', 'r') as f:
        train_data = json.load(f)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data, tokenizer=tokenizer,
        max_seq_length=2048,
        # wrap in HuggingFace Dataset
        )
    trainer.train()
    model.save_pretrained('./checkpoints/hindi_lora/final')

if __name__ == "__main__":
    train_llama_backbone()