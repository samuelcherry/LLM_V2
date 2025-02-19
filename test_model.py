import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

input_text = "The workers role is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\nGenerated text:\n", output_text)