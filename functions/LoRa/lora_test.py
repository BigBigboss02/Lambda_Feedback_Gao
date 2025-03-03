from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel
import torch
import pandas as pd
from langchain.prompts import PromptTemplate

# Define paths

adapter_folder = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/23022025"
base_model_name = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/Llama-3.2-1B"  # Change this to match the model the adapter was trained on
csv_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/semantic_comparisons_lower_pressure.csv"
# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Use MPS if available

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
base_model = LlamaForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype=torch.float16, 
    device_map={"": "mps"}  # Use "cpu" if "mps" doesn't work
)
# Load adapter model
model = PeftModel.from_pretrained(base_model, adapter_folder).to(device)
# model = base_model
model.eval()

import re

def extract_last_boolean(llm_output):
    if not llm_output or not isinstance(llm_output, list) or not llm_output[0]:
        return 'False'
    text = llm_output[0]
    # Search for the last occurrence of 'True' or 'False' (case insensitive)
    match = list(re.finditer(r'\b(true|false)\b', text, re.IGNORECASE))
    if not match:
        return 'False'
    return match[-1].group(1).title()  # Return the last match, normalized to 'True' or 'False'
# Test inference
prompt_template = prompt_template = PromptTemplate(
    template='''
    <s>[INST]
    Determine if the 2 words are semantically similar. Provide 'True' or 'False'. Word1:{target}, Word2:{word}
    [/INST]
    ''',
    input_variables=["target", "word"]
)

input_text = prompt_template.format(target="pressure", word="dsbahjdas")
print(input_text)
inputs = tokenizer(input_text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to the same device as the model

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)  # Use generate

# Decode and print model output
decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
cleaned_output = extract_last_boolean([text.strip() for text in decoded_output])  # Remove whitespace
print(cleaned_output)



df = pd.read_csv(csv_path, names=["Word1", "Word2", "Ground Truth"])
# Store results
results = []

for _, row in df.iterrows():
    input_text = prompt_template.format(target=row["Word1"], word=row["Word2"])
    print(input_text)
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to the same device as the model

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)  # Use generate

    # Decode and print model output
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned_output = [text.strip() for text in decoded_output]
    #cleaned_output = extract_last_boolean([text.strip() for text in decoded_output])  # Remove whitespace
    print(cleaned_output)
    results.append([row["Word1"], row["Word2"], row["Ground Truth"], cleaned_output])

# Convert results to a DataFrame
output_df = pd.DataFrame(results, columns=["Word1", "Word2", "Ground Truth", "Model Output"])

# Save to CSV
output_df.to_csv("/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/finetuned_model_test/output_results_lora_full.csv", index=False)

print("Inference complete. Results saved to output_results.csv.")