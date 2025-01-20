import pandas as pd
import json

# File paths
json_file_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/pairs.json"
csv_file_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/llm_results_instructions.csv"
output_file_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/matched_results_instructions.csv"

# Load JSON data
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

# Convert JSON data to a dictionary for quick lookup
json_lookup = {
    (entry[0], entry[1]): entry[2] for entry in [eval(item) for item in json_data]
}

# Load CSV data
df = pd.read_csv(csv_file_path)

# Ensure CSV has the expected columns
df.columns = [col.strip().lower() for col in df.columns]
assert 'target' in df.columns and 'word' in df.columns and 'response' in df.columns, "CSV must contain 'target', 'word', and 'response' columns"

# Create a new column for match result
def match_result(row):
    json_response = json_lookup.get((row['target'], row['word']), None)
    csv_response = row['response'].strip()

    if not csv_response:  # If CSV response is empty
        return "nothing" if json_response is None else "false"

    if csv_response.lower() == json_response:
        return "true"

    if json_response == "unsure" and csv_response.lower() not in ["true", "false"]:
        return "true"

    return "false"

df["match_result"] = df.apply(match_result, axis=1)

# Save the updated DataFrame to a new CSV
df.to_csv(output_file_path, index=False)

print(f"Match results saved to {output_file_path}")