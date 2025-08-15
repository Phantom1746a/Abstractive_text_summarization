# train.py
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch
import os

# ================================
# 1. Load Dataset from CSV
# ================================
dataset = load_dataset("csv", data_files="data/urdu_summaries.csv")["train"]

# Clean: Remove empty or short entries
def clean_data(example):
    return (
        example["text"] and example["summary"] and
        len(example["text"].strip()) > 10 and
        len(example["summary"].strip()) > 5
    )

dataset = dataset.filter(clean_data)

print(f"Loaded {len(dataset)} examples.")


# ================================
# 2. Format for Instruction Tuning
# ================================
def formatting_prompts_func(examples):
    instructions = examples["text"]
    responses = examples["summary"]
    texts = [
        f"""ذیل کے متن کا خلاصہ کریں:
{instruction}

خلاصہ: {response}"""
        for instruction, response in zip(instructions, responses)
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=2)


# ================================
# 3. Load Model & Tokenizer with Unsloth
# ================================
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-v0.1",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


# ================================
# 4. Set Up Trainer
# ================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_urdu_mistral_lora",
        save_strategy="steps",
        save_steps=25,
        report_to="none",
        per_device_eval_batch_size=1,
    ),
)


# ================================
# 5. Start Training
# ================================
print("Starting training...")
trainer.train()


# ================================
# 6. Save LoRA Adapter
# ================================
model.save_pretrained("mistral-7b-urdu-sum-lora")
tokenizer.save_pretrained("mistral-7b-urdu-sum-lora")

print("✅ Model saved to 'mistral-7b-urdu-sum-lora'")