from omni_speech.fine_tuning.lora_config import lora_config
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
import json, torch



def train_llama_backbone():
    model, tokenizer = lora_config()

    
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
        eval_steps=200,
        dataloader_num_workers=4,
        report_to='tensorboard',
        gradient_checkpointing=True, # effective batch = 16
        optim='adamw_torch',
        max_grad_norm=1.0,
        )
    
    if training_args.gradient_checkpointing:
        #* the trainer usually enable it by defualt but with lora it's safer todo this
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    # Load your prepared data
    with open('data/hindi_dataset_backbone.json', 'r', encoding='utf-8') as f: #! why utf-8? for loading of hindi text
        train_data = json.load(f)

    # making it into a HuggingFace Dataset
    train_data = Dataset.from_list(train_data)

    def process_data(examples):
        text = tokenizer.apply_chat_template(examples["messages"], tokenize=False)
        return {
            "text": text,
        }
    
    train_data = train_data.map(process_data, remove_columns=["messages"])
    
    
   # print(train_data[0])


    trainer = SFTTrainer(
        model=model.to("cuda"),
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer,
        max_seq_length=2048,
        dataset_text_field="text",
        # wrap in HuggingFace Dataset
        )
    trainer.train()
    model.save_pretrained('./checkpoints/hindi_lora/final')

if __name__ == "__main__":
    
    
    train_llama_backbone()