import os
import re
import time
import pandas as pd
import openai
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class Param:
    def __init__(self, model='gpt-4o-mini', temperature=0.01, max_tokens=5):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens


def parse_last_boolean(response):
    matches = re.findall(r'\b(true|false)\b', response, re.IGNORECASE)
    return matches[-1].lower() if matches else "unsure"


def call_openai_api(prompt, param: Param):
    try:
        response = openai.ChatCompletion.create(
            model=param.model,
            temperature=param.temperature,
            max_tokens=param.max_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"API Error: {str(e)}"


def build_prompt(word1, word2):
    return f"""
### Instruction:
Determine if the 2 words are semantically similar. Provide one of the following responses:
- "True" if the words are semantically the same.
- "False" if the words are semantically different.

### Examples:
Word1: "velocity", Word2: "speed"  
Response: True

Word1: "neuron", Word2: "planet"  
Response: False

### Input:
Word1: {word1}, Word2: {word2}

### Response:
"""


def split_text_column(text):
    match = re.match(r"Word1:\s*(.*?)\s+Word2:\s*(.*)", text)
    if match:
        return match.group(1), match.group(2)
    else:
        return "", ""


def main():
    input_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/data/minimum_logic_trainining_dataset_vs2.csv"
    output_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/one_to_one_tests/initial_1to1_tests/gpt4omini_1to1.csv"

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    param = Param()
    results = []

    for _, row in df.iterrows():
        word1, word2 = split_text_column(row['text'])
        label = str(row['label']).strip().lower()

        prompt = build_prompt(word1, word2)
        response = call_openai_api(prompt, param)
        predicted = parse_last_boolean(response)

        is_correct = predicted == label
        results.append({
            "Word1": word1,
            "Word2": word2,
            "Ground Truth": label,
            "Model Output": predicted,
            "Correct?": is_correct,
            "Raw Response": response
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
