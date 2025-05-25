import os
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import pandas as pd
from langchain.prompts import PromptTemplate

# --- Config ---
base_model_name = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/base_models/Llama-3.2-1B"
csv_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/data/minimum_logic_testing_dataset.csv"
output_base_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/one_to_one_tests/Base_controlled_variable_tests/llama32_1b"
limit_rows = False  # Set to True to test only 50 rows

# --- Device setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = LlamaForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map={"": "mps"}
)

# --- Prompt templates ---
prompting_variants = {
    "null": PromptTemplate(
        template="Word1:{target}, Word2:{word}",
        input_variables=["target", "word"]
    ),
    "simple_instruction": PromptTemplate(
        template="<s>[INST] Determine if the 2 words are semantically similar. Provide 'True' or 'False'. [/INST] Word1:{target}, Word2:{word} [/INST]",
        input_variables=["target", "word"]
    ),
    "examples_only": PromptTemplate(
        template=(
            "<s>[INST] Examples:\n"
            "Word1: color, Word2: colour [/INST] True\n"
            "Word1: happy, Word2: sad [/INST] False\n"
            "Word1: red, Word2: Red [/INST] True\n"
            "Word1:{target}, Word2:{word} [/INST]"
        ),
        input_variables=["target", "word"]
    ),
    "instruction_and_examples": PromptTemplate(
        template=(
            "<s>[INST] Determine if the 2 words are semantically similar. Provide 'True' or 'False'.\n"
            "Examples:\n"
            "Word1: color, Word2: colour [/INST] True\n"
            "Word1: happy, Word2: sad [/INST] False\n"
            "Word1: red, Word2: Red [/INST] True\n"
            "Word1:{target}, Word2:{word} [/INST]"
        ),
        input_variables=["target", "word"]
    )
}

# --- Load and process dataset ---
df = pd.read_csv(csv_path)
df[['Word1', 'Word2']] = df['text'].str.extract(r'Word1:\s*(\S+)\s*Word2:\s*(\S+)', expand=True)
df = df.drop(columns=["text"])
if limit_rows:
    df = df.sample(n=50, random_state=42).reset_index(drop=True)

# --- Inference for each prompt strategy ---
model = base_model.to(device)
model.eval()

for prompt_name, prompt_template in prompting_variants.items():
    print(f"\n--- Running inference for prompt: {prompt_name} ---")
    results = []

    for _, row in df.iterrows():
        input_text = prompt_template.format(target=row["Word1"], word=row["Word2"])
        print(f"Input Text: {input_text.strip()}")

        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                temperature=0.01
            )

        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(f"Model Output: {decoded_output}")

        # Directly save raw decoded output
        results.append([row["Word1"], row["Word2"], row["label"], decoded_output])

    # Save result with original column names
    output_df = pd.DataFrame(results, columns=["Word1", "Word2", "Ground Truth", "Model Output"])
    output_path = os.path.join(output_base_path, f"{prompt_name}_prompt_output.csv")
    output_df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")