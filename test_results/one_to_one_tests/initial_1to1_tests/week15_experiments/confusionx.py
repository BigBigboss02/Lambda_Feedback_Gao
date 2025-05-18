import pandas as pd
import json

# File paths

output_file_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/matched_results.csv"
import pandas as pd

# Load the uploaded files
file1_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/json_pairs_cleaned.csv"
file2_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/llm_results_instructions.csv"

# Reading the CSV files
file1 = pd.read_csv(file1_path)
file2 = pd.read_csv(file2_path)

# Merge the two files based on the first two columns ('target' and 'word')
merged = pd.merge(file1, file2, on=['target', 'word'], suffixes=('_file1', '_file2'))

# Define a function to handle the comparison logic
def compare_responses(response1, response2):
    response1 = response1.strip().lower()
    response2 = response2.strip().lower()
    
    # Match logic
    if response1 in ['true', 'false']:
        if response1 == response2:
            return True
        else:
            return False
    # Unsure l
    elif response1 not in ['true', 'false'] and response2 == 'unsure':
        return True
    else:
        return False

# Apply the comparison function
merged['comparison_result'] = merged.apply(
    lambda row: compare_responses(row['response_file1'], row['response_file2']), axis=1
)

# Select relevant columns for output
result = merged[['target', 'word', 'response_file1', 'response_file2', 'comparison_result']]
print(result)