import pandas as pd

# Load the CSV file to inspect its structure
file_path = r'C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\result\Cross_Platform_Comparison\Llama3_1B_test\trial1.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
data.head()

# Filter rows with 'Incorrect' in the Response column
incorrect_count = data['Response'].apply(lambda x: 'Incorrect' in x).sum()

# Calculate total rows
total_count = len(data)

# Calculate percentage of 'Incorrect'
percentage_incorrect = (incorrect_count / total_count) * 100

print(f"The percentage of 'Incorrect' responses is {percentage_incorrect:.2f}%.")
