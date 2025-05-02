from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from datasets import load_dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "dataset.txt"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "train" / "factbot"
LOGS_DIR = PROJECT_ROOT / "train" / "logs"

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

try:
    dataset = load_dataset("text", data_files={"train": str(DATASET_PATH)})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        per_device_train_batch_size=4,
        num_train_epochs=10,
        logging_steps=10,
        save_steps=100,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(LOGS_DIR),
        save_total_limit=2,
        learning_rate=5e-5,
        gradient_accumulation_steps=4,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

except FileNotFoundError as e:
    print(f"Error: Dataset file not found at {DATASET_PATH}")
except Exception as e:
    print(f"Error during training: {str(e)}")







