from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel
import torch

# Define paths

adapter_folder = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/23022025"
base_model_name = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/Llama-3.2-1B"  # Change this to match the model the adapter was trained on

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Use MPS if available

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = LlamaForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype=torch.float16, 
    device_map={"": "mps"}  # Use "cpu" if "mps" doesn't work
)
# Load adapter model
model = PeftModel.from_pretrained(base_model, adapter_folder).to(device)
model.eval()

# Test inference
input_text = '''
### Instruction:
Determine if the 2 words are semantically similar. Provide 'True' or 'False'
### Input:
Word1: Pressure, Word2: Energy
'''
inputs = tokenizer(input_text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to the same device as the model

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)  # Use generate

# Decode and print model output
decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
cleaned_output = [text.strip() for text in decoded_output]  # Remove whitespace
print(cleaned_output)