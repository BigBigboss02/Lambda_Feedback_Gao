import pandas as pd

# Load the CSV file
file_path = r'C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\result\Cross_Platform_Comparison\GPT-4o_mini_8B_test\trial2.csv'
data = pd.read_csv(file_path)

# Print column names for debugging
print("Columns in the dataset:", data.columns)

# Clean column names
data.columns = data.columns.str.strip()

# Ensure the 'Assistant Output' column exists
if 'Assistant Output' in data.columns:
    # Strip extra spaces from the 'Assistant Output' column
    data['Assistant Output'] = data['Assistant Output'].str.strip()

    # Count the total number of rows
    total_rows = len(data)

    # Count the number of rows where 'Assistant Output' is 'Output: Incorrect'
    incorrect_rows = data['Assistant Output'].value_counts().get('Output: Incorrect', 0)

    # Calculate the percentage of 'Incorrect' responses
    percentage_incorrect = (incorrect_rows / total_rows) * 100

    print(f"The percentage of 'Incorrect' responses is: {percentage_incorrect:.2f}%")
else:
    print("The column 'Assistant Output' does not exist in the dataset.")
