import json
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Load your 20‑example dataset
with open("human_dataset.json", "r") as f:
    examples = json.load(f)

def format_example(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Report:\n{example['input']}\n\n### Patient Explanation:\n{example['output']}"

texts = [format_example(ex) for ex in examples]
dataset = Dataset.from_dict({"text": texts})

# Load base model in 4‑bit
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Attach LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=100,               # 100 steps on 20 examples
    learning_rate=2e-4,
    bf16=True,                   # AMD MI300X supports bfloat16
    logging_steps=5,
    output_dir="pathology_model_output",
    save_steps=50,
    report_to="none",
)

# SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048,
)

print("Starting fine‑tuning...")
trainer.train()

# Save the LoRA adapter
model.save_pretrained("pathology_finetuned")
tokenizer.save_pretrained("pathology_finetuned")
print("✅ Fine‑tuned model saved to 'pathology_finetuned/'")