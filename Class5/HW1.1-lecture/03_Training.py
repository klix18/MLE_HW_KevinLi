# ---- Setup base model + LoRA ----
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset
import torch

model_name = "microsoft/DialoGPT-small"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 family has no pad_token

# Base model (CPU/FP32 by default; will move to GPU if available)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

# LoRA config (target GPT-2 attention projections)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,   # we're training
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]
)

# Wrap with PEFT/LoRA
lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()

# (Optional) move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
lora_model.to(device)

# ---- Tiny demo dataset ----
demo_dataset = Dataset.from_list([
    {"text": "Human: What is Python?\nAssistant: Python is a programming language."},
    {"text": "Human: How do I learn coding?\nAssistant: Start with basic concepts and practice regularly."}
])

# ---- Training args ----
training_args = TrainingArguments(
    output_dir="./trl_sft_demo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=5e-4,
    logging_steps=1,
    save_steps=10,
    save_total_limit=1,
    fp16=False,          # set True if using a GPU that supports FP16
    report_to=None,      # no W&B/etc.
)

# ---- Trainer ----
trainer = SFTTrainer(
    model=lora_model,
    args=training_args,
    train_dataset=demo_dataset,
    tokenizer=tokenizer,          # good to pass tokenizer (padding/formatting)
    formatting_func=lambda ex: ex["text"]  # TRL expects text; this maps rows to strings
)

# ---- Train ----
trainer.train()

# ---- Save LoRA adapter (and training args) ----
trainer.save_model("./trl_sft_demo/adapter")   # saves only the LoRA weights
print("✅ Saved LoRA adapter to ./trl_sft_demo/adapter")
