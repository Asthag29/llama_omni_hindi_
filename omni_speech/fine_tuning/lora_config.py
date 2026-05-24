
from peft import LoraConfig, get_peft_model, TaskType
from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments
import torch, json

def lora_config():

    #!loading the backbone
    model_path = "./models/llama"

    #!automodel for causal lm reads the model architecture from the config file
    model = LlamaForCausalLM.from_pretrained(
    model_path, 

    ignore_mismatched_sizes=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
     )
    model.to("cuda")
    #! tokenizer reads the tokenizer configuration 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

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
    model.config.use_cache = False

    #* required for the gradient flowing since the wwights are frozen and lora weights can't be trained without this
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    return model, tokenizer