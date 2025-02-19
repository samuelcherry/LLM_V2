import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling

print("CUDA available:", torch.cuda.is_available())
print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

def chunk_text(text,tokenizer, max_length=1024, stride=512):
    tokenizer_text = tokenizer(text, truncation=False, padding=False, max_length=max_length)
    tokens = tokenizer_text['input_ids']

    chunks = []
    for i in range (0, len(tokens), max_length - stride):
        chunk = tokens[i:i + max_length]
        chunks.append(chunk)

    return chunks


def load_dataset(file_path, tokenizer, max_length=1024, stride=512):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = chunk_text(text, tokenizer, max_length, stride)

    tokenized_chunks = []
    for chunk in chunks:
        padded_chunk = chunk + [tokenizer.pad_token_id] * (max_length - len(chunk))
        tokenized_chunks.append({
            "input_ids": padded_chunk,
            "labels": padded_chunk
        })
    
    return tokenized_chunks

train_dataset = load_dataset("combined_books.txt", tokenizer)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./trained_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

print("Training complete! Model saved in './trained_model'")