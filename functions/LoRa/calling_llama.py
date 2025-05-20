from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel
import torch
import pandas as pd
import re
from langchain.prompts import PromptTemplate

# Define paths
adapter_folder = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/controlled_variable/1603examples_logic_newarg"
base_model_name = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/Llama-3.2-1B"
csv_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/minimum_logic_dataset_test.csv"

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is eos token

base_model = LlamaForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype=torch.float16, 
    device_map={"": "mps"}
)

# Load adapter model
model = PeftModel.from_pretrained(base_model, adapter_folder).to(device)
model.eval()

def extract_last_boolean(llm_output):
    if not llm_output or not isinstance(llm_output, list) or not llm_output[0]:
        return 'False'
    text = llm_output[0]
    match = list(re.finditer(r'\b(true|false)\b', text, re.IGNORECASE))
    return match[-1].group(1).title() if match else 'False'

# Prompt template
prompt_template = PromptTemplate(
    template='''
    <s>[INST]
    Determine if the 2 words are semantically similar. Provide 'True' or 'False'. Word1:{target}, Word2:{word}
    [/INST]
    ''',
    input_variables=["target", "word"]
)

# Load and preprocess DataFrame
df = pd.read_csv(csv_path)
print("Original DataFrame:")
print(df.head())

# Extract Word1 and Word2 from 'text' column
df[['Word1', 'Word2']] = df['text'].str.extract(r'Word1:\s*(\S+)\s*Word2:\s*(\S+)')
df = df.drop(columns=["Unnamed: 0", "text"])  # Drop unnecessary columns if needed
print("Processed DataFrame:")
print(df.head())

# Store results
results = []

for _, row in df.iterrows():
    input_text = prompt_template.format(target=row["Word1"], word=row["Word2"])
    print(f"Input Text: {input_text}")
    
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to device

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)

    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned_output = extract_last_boolean(decoded_output)
    
    print(f"Model Output: {cleaned_output}")

    results.append([row["Word1"], row["Word2"], row["label"], cleaned_output])

# Convert results to DataFrame
output_df = pd.DataFrame(results, columns=["Word1", "Word2", "Ground Truth", "Model Output"])

# Save results
output_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/finetuned_model_test/controlled_variable/1603examples_logic_newarg.csv"
output_df.to_csv(output_path, index=False)

print("Inference complete. Results saved to:", output_path)