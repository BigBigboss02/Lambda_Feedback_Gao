from datasets import Dataset
import pandas as pd
from huggingface_hub import login

login() # You will need to enter your Hugging Face token.

# 2️⃣ Load the structured dataset
csv_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/minimum_logic_dataset.csv"  # Make sure this CSV is saved after processing
df = pd.read_csv(csv_path)
# Combine 'instruction' and 'input' into a single column named 'input'
df['input'] = df['text']
df['response'] = df['label']
# Keep only 'input' and 'response' columns
df = df[['input', 'response']]


# 3️⃣ Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# 4️⃣ Push to Hugging Face Hub
dataset.push_to_hub("Bigbigboss02/minimum_logic_200x5")

print("✅ Dataset successfully uploaded!")