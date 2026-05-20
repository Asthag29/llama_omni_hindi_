
from peft import LoraConfig, get_peft_model, TaskType
from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments
import torch, json

def lora_config():

    #!loading the backbone
    model_path = "./models/llama"

    #!automodel for causal lm reads the model architecture from the config file
    model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto",
    ignore_mismatched_sizes=True,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
     )

    #! tokenizer reads the tokenizer configuration 
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    #* LoRA configuration
    lora_config = LoraConfig(

        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    #* get the peft model(parameter efficient fine-tuning)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer