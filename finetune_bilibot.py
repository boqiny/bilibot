from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2-1.5b-bnb-4bit", # Reminder we support ANY Hugging Face model!
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

alpaca_prompt = """
你是一位B站老用户，请使用暴躁的语言风格，对以下问题给出简短、机智的回答：

### 用户:
{}

### 输出:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    inputs = examples["source"]
    outputs = examples["target"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("ZheYiShuHua/bilibot", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# Split the dataset
total_size = len(dataset)
val_size = 500
train_size = total_size - val_size

# Create train and validation splits
train_dataset = dataset.select(range(val_size, total_size))
val_dataset = dataset.select(range(val_size))

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=18,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        warmup_ratio=0.1,
        learning_rate=3e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="bilibot_outputs",

        fp16_full_eval=True,
        per_device_eval_batch_size=18,
        eval_accumulation_steps=2,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        report_to="tensorboard",
        logging_dir="bilibot_outputs/logs",
    ),
)

print("Starting training...")
trainer_stats = trainer.train()

print("Saving model and tokenizer...")
model.save_pretrained("bilibot_outputs/lora_model")
tokenizer.save_pretrained("bilibot_outputs/lora_model")