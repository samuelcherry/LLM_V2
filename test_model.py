import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyttsx3

engine =pyttsx3.init()

engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

model_path = "./trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

input_text = "Once there was a hero,"
inputs = tokenizer(input_text, return_tensors="pt", padding=True)
input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)

tokenizer.pad_token = tokenizer.eos_token 
model.config.pad_token_id = tokenizer.eos_token_id

output_ids = model.generate(
    input_ids,
    max_length=20,
    do_sample=True,
    top_k=50,
    top_p=0.92,
    temperature=0.8,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    attention_mask=attention_mask,
)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)