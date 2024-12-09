import requests
import json
from dotenv import load_dotenv
import os
import time

# Specify the path to the renamed .env file
env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
load_dotenv(dotenv_path=env_path)
  

# Configuration class to load environment variables
class Config():
    API_URL = os.getenv("OPENAI_URL")  # Replace with the correct key for GPT-4-O API URL
    AUTHORIZATION = os.getenv("OPENAI_API_KEY")  # Replace with the correct authorization key variable
    examples_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\functions\python_script\structured_prompts\examples1.json"
    csv_saving_basepath = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\test_results\confusion_matrix\confusion_matrix"

Config = Config()

API_URL = Config.API_URL
headers = {"Authorization": f"Bearer {Config.AUTHORIZATION}"}






import json
from langchain.prompts import PromptTemplate

# Define the prompt template
template_text = """
Below is an instruction to determine right or wrong on student's coursework answers, paired with an input that provides further context. Give binary response that appropriately completes the request.

### Instruction:
You are checking if the input includes 3 similar answers from the given list:
{correct_answers}

### Examples:
{examples_text}

### Input:
{test}
"""
prompt_template = PromptTemplate(
    template=template_text,
    input_variables=["correct_answers", "examples_text", "test"]
)

# Load JSON data for correct answers
with open(r'Lambda_Feedback_Gao\functions\python_script\structured_prompts\confusion_matrix\A_Level_STEM_Answers.json', "r") as file:
    correct_answers_temp = json.load(file)
with open(r"Lambda_Feedback_Gao\functions\python_script\structured_prompts\confusion_matrix\A_Level_STEM_Examples.json", "r") as file:
    examples_temp = json.load(file)


correct_answers = "\n".join(correct_answers_temp["example_text"])  # Join list elements
# Extract and format examples
if "examples_with_correctness" in examples_temp:
    all_examples = []
    for subject, examples in examples_temp["examples_with_correctness"].items():
        for example in examples:
            all_examples.append(
                f"Input: {example['input']}\nOutput: {'Correct' if example['correct'] else 'Incorrect'}"
            )
    examples_data = "\n".join(all_examples)

# Iterate through test examples and compare results
results = []
for subject, examples in examples_temp["examples_with_correctness"].items():
    for example in examples:
        # Build the full prompt
        full_prompt = prompt_template.format(
            correct_answers=correct_answers,
            examples_text=examples_data,
            test=example['input']
        )

        # Prepare the payload
        payload = {
            "model": "gpt-4o-mini",  # Specify GPT-4-O Mini as the model
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": 100,  # Limit response length
            "temperature": 0.7  # Control randomness
        }

        # Simulate model response (replace this with actual API call)
        # For now, assume the model outputs 'Correct' or 'Incorrect'
        model_response = "Correct" if example["correct"] else "Incorrect"

        # Compare with expected output
        is_correct = (model_response == ("Correct" if example["correct"] else "Incorrect"))
        results.append({
            "input": example["input"],
            "expected": "Correct" if example["correct"] else "Incorrect",
            "output": model_response,
            "result": "Pass" if is_correct else "Fail"
        })

# Output results
for result in results:
    print(f"Input: {result['input']}")
    print(f"Expected: {result['expected']}")
    print(f"Output: {result['output']}")
    print(f"Result: {result['result']}")
    print("---")



import csv
from datetime import datetime

# Define the CSV saving base path
csv_saving_basepath = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\functions\python_script\structured_prompts\confusion_matrix"

# Generate a unique file name using the current timestamp
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filepath = os.path.join(csv_saving_basepath, f"results_{current_time}.csv")

# Initialize the CSV file with a header row
with open(filepath, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Input", "Expected Output", "Model Output", "Result"])  # Define column headers

# Write results to the CSV file
with open(filepath, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Iterate through results and write each as a row
    for result in results:
        writer.writerow([result["input"], result["expected"], result["output"], result["result"]])

print(f"Results have been saved to {filepath}")
# import csv
# from datetime import datetime
# # Update the query function to extract only relevant response data
# def query_with_wait(api_url, headers, payload, initial_wait_time=5, retry_interval=60, max_retries=5):
#     print(f"Waiting {initial_wait_time} seconds for the endpoint to load...")
#     time.sleep(initial_wait_time)

#     for attempt in range(max_retries):
#         response = requests.post(api_url, headers=headers, json=payload)

#         if response.status_code == 200:
#             # Extract the assistant's message content
#             response_data = response.json()
#             choices = response_data.get("choices", [])
#             if choices:
#                 assistant_message = choices[0]["message"]["content"]
#                 return assistant_message  # Return only the assistant's message
#             else:
#                 print("No valid choices in response.")
#                 return None
#         elif response.status_code == 503:  # Service Unavailable
#             print(f"Attempt {attempt + 1}/{max_retries}: Model is still loading. Retrying in {retry_interval} seconds...")
#             time.sleep(retry_interval)
#         else:
#             print(f"Error: {response.status_code}, {response.text}")
#             break
#     else:
#         print("Failed to get a response after multiple retries.")
#         return None





# # Initialize the CSV file
# current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# filepath = os.path.join(Config.csv_saving_basepath, f"{current_time}.csv")
# # Write the header row
# with open(filepath, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Query Number", "Timestamp", "Assistant Output"])  # Update column name for clarity

# # Perform multiple queries
# for i in range(5):  # Adjust the range for the desired number of queries
#     print(f"Performing query {i + 1}")
#     output = query_with_wait(API_URL, headers, payload)

#     if output:
#         # Save the extracted assistant's output to CSV
#         with open(filepath, mode='a', newline='', encoding='utf-8') as file:
#             writer = csv.writer(file)
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             writer.writerow([i + 1, timestamp, output])  # Log only the assistant's output
#     else:
#         print(f"Query {i + 1} failed.")