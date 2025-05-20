from datasets import Dataset
import pandas as pd
from huggingface_hub import login


with_instruction = True
if_push = True

if if_push:
    login() # You will need to enter your Hugging Face token.

# 2️⃣ Load the structured dataset
csv_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/database/minimum_logic_dataset.csv"  # Make sure this CSV is saved after processing
df = pd.read_csv(csv_path)


# Define diverse examples for context
examples = """
[/INST]\n Examples: 
Word1: color, Word2: colour  [/INST]\n True
Word1: happy, Word2: sad  [/INST]\n False
Word1: red, Word2: Red  [/INST]\n True
"""

# with instruction and examples
df['text'] = df['text'].apply(lambda x: f"Determine if the 2 words are semantically similar. Provide 'True' or 'False'. {examples}{x}")

#  Examples only
df['text'] = df['text'].apply(lambda x: f"{examples}{x}")

if with_instruction:
    # Modify the 'text' column to the new format
    df['text'] = df['text'].apply(lambda x: f"Determine if the 2 words are semantically similar. Provide 'True' or 'False'. {x}")

# Combine 'instruction' and 'input' into a single column named 'input'
df['input'] = df['text']
df['response'] = df['label']
# Keep only 'input' and 'response' columns
df = df[['input', 'response']]

pd.set_option('display.max_colwidth', None)  # Show full content
print(df.iloc[0])  # Print the first row # Display the first row with all its values

if if_push:
    # 3️⃣ Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    # 4️⃣ Push to Hugging Face Hub
    dataset.push_to_hub("Bigbigboss02/instructive_examples_logic_200x5")

    print("✅ Dataset successfully uploaded!")