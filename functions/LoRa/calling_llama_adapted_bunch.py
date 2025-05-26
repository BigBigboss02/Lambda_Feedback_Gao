import os
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel
import torch
import pandas as pd
import re
from langchain.prompts import PromptTemplate

# Define paths
base_model_name = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/base_models/Llama-3.2-1B"
csv_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/data/minimum_logic_testing_dataset_vs2.csv"
adapters_folder = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/tuned_Llama321B_balanced_dataset"
output_base_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/one_to_one_tests/LoRa_controlled_variable_tests/Llama3-1B_vs2/adaptervs2/instructive_examples_prompt"
limit_rows = False  # Set to True to limit the number of rows for testing

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is eos token

# base_model = LlamaForCausalLM.from_pretrained(
#     base_model_name, 
#     torch_dtype=torch.float16, 
#     device_map={"": "mps"}
# )
base_model = LlamaForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16
).to(device)
# prompt_template = PromptTemplate(
#     template='''
#     <s>[INST]\n 
#     Word1:{target}, Word2:{word}
#     [/INST]
#     ''',
#     input_variables=["target", "word"]
# )
# Prompt template
# prompt_template = PromptTemplate(
#     template='''
#     <s>[INST]
#     Determine if the 2 words are semantically similar. Provide 'True' or 'False'. 
#     \n[/INST]\n 
#     Word1:{target}, Word2:{word}
#     [/INST]
#     ''',
#     input_variables=["target", "word"]
# )

# prompt_template = PromptTemplate(
#     template='''
#     <s>[INST]
#     Examples: 
#     Word1: color, Word2: colour  [/INST]\n True
#     Word1: happy, Word2: sad  [/INST]\n False
#     Word1: red, Word2: Red  [/INST]\n True
#     \n[/INST]\n 
#     Word1:{target}, Word2:{word}
#     [/INST]
#     ''',
#     input_variables=["target", "word"]
# )

#with instruction and examples
prompt_template = PromptTemplate(
    template='''
    <s>[INST]
    Determine if the 2 words are semantically similar. Provide 'True' or 'False'. 
    \n[/INST]\n 
    Examples: 
    Word1: color, Word2: colour  [/INST]\n True
    Word1: happy, Word2: sad  [/INST]\n False
    Word1: red, Word2: Red  [/INST]\n True
    \n[/INST]\n 
    Word1:{target}, Word2:{word}
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
df = df.drop(columns=["text"])  # Drop unnecessary columns if needed
print("Processed DataFrame:")
print(df.head())


# Randomly select 50 rows and drop the rest
if limit_rows:
    df = df.sample(n=50, random_state=42).reset_index(drop=True)

# Loop through all adapter folders in `adapters_folder`
for adapter_name in os.listdir(adapters_folder):
    adapter_path = os.path.join(adapters_folder, adapter_name)
    
    if not os.path.isdir(adapter_path):  # Skip non-folder items
        continue

    print(f"\nProcessing adapter: {adapter_name}")

    # Load adapter model
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    model.eval()

    # Store results
    results = []

    for _, row in df.iterrows():
        input_text = prompt_template.format(target=row["Word1"], word=row["Word2"])
        print(f"Input Text: {input_text}")

        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to device

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False,temperature = 0.01)#using greedy decoding

        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned_output = decoded_output
        print(f"Model Output: {cleaned_output}")

        results.append([row["Word1"], row["Word2"], row["label"], cleaned_output])

    # Convert results to DataFrame
    output_df = pd.DataFrame(results, columns=["Word1", "Word2", "Ground Truth", "Model Output"])

    # Save results using the adapter folder name as part of the filename
    output_path = os.path.join(output_base_path, f"{adapter_name}.csv")
    output_df.to_csv(output_path, index=False)

    print(f"Inference complete for {adapter_name}. Results saved to: {output_path}")