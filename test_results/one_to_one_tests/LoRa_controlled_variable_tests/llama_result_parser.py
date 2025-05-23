import os
import pandas as pd

# === File Paths ===
source_folder = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/one_to_one_tests/LoRa_controlled_variable_tests/Llama3-1B/instructive_examples_prompt"
destination_folder = os.path.join(source_folder, "parsed_results")
os.makedirs(destination_folder, exist_ok=True)

# === Brutally Simple Parser ===
def detect_last_word_truth_value(text, row_index=None):
    if not isinstance(text, str):
        return "ambiguous"

    # Clean list-style wrapping and double quotes
    if text.startswith('["') and text.endswith('"]'):
        text = text[2:-2]
    text = text.replace('""', '"').strip()

    # Get last word (after whitespace split)
    words = text.strip().split()
    last_word = words[-1].lower() if words else ""

    if row_index is not None and row_index < 10:
        print(f"\n🔍 Row {row_index} | Last word: {repr(last_word)}")

    if last_word == "true":
        return "true"
    elif last_word == "false":
        return "false"
    else:
        return "ambiguous"

# === File Processing Loop ===
for filename in os.listdir(source_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(source_folder, filename)
        df = pd.read_csv(file_path)

        if len(df.columns) >= 4:
            print(f"\n🗂 Processing: {filename}")
            model_output_column = df.columns[3]  # 4th column
            df["parsed_truth_value"] = [
                detect_last_word_truth_value(row, i)
                for i, row in enumerate(df[model_output_column])
            ]
        else:
            print(f"❌ Skipped {filename}: not enough columns")

        # Save updated file
        output_path = os.path.join(destination_folder, filename)
        df.to_csv(output_path, index=False)