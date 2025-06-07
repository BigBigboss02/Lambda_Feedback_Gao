import requests
import json
from dotenv import load_dotenv
import os
import time
from datetime import datetime
import csv

# Specify the path to the renamed .env file
env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\environments\login_configs.env"
load_dotenv(dotenv_path=env_path)

class Config():
    API_URL = os.getenv("3B_API_URL")
    AUTHORIZATION = os.getenv("HUGGINGFACE_AUTHORIZATION")
    examples_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\functions\python_script\structured_prompts\examples1.json"   
Config = Config()

API_URL = Config.API_URL
headers = {"Authorization": f"Bearer {Config.AUTHORIZATION}"}

# Define the list of WSN applications
prompt = ('''
Environmental monitoring
Health care monitoring
Industrial process control
Smart agriculture
Structural health monitoring
Habitat monitoring
Disaster response systems
Military surveillance
Wildlife tracking
Smart homes
Traffic monitoring
Environmental pollution tracking
Oil and gas pipeline monitoring
Water quality monitoring
Precision agriculture
Forest fire detection
Urban planning
Air quality monitoring
Smart grid management
Waste management systems
Flood detection
Building security systems
Patient health monitoring
Crop yield monitoring
Weather forecasting
Power line monitoring
Road condition monitoring
Asset tracking
Energy usage monitoring
Smart parking systems
''')

# Load examples from JSON file
with open(Config.examples_path, "r") as file:
    examples = json.load(file)

# Format examples as part of the prompt
examples_text = "\n".join(
    [f"Input: {item['original']}\nOutput: {'Correct' if item['correct'] else 'Incorrect'}\n"
     for item in examples]
)
print(examples_text)

# Sample input data
test = "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. KFC takeaway, 2. Energy usage monitoring, 3. Smart parking systems."

# Build the full prompt
full_prompt = f"""
Below is an instruction to determine right or wrong on student's coursework answers, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are checking if the input includes 3 valid Wireless Sensor Network (WSN) applications from the given list:
{prompt}

### Examples:
{examples_text}

### Input:
{test}
"""

# Prepare the payload
payload = {
    "inputs": full_prompt,
    "parameters": {
        "max_new_tokens": 100,  # Limit response length
        "temperature": 0.7,     # Control randomness
    }
}

# Function to query the API with retry logic
def query_with_wait(api_url, headers, payload, initial_wait_time=257, retry_interval=60, max_retries=5):
    print(f"Waiting {initial_wait_time} seconds for the endpoint to load...")
    time.sleep(initial_wait_time)

    for attempt in range(max_retries):
        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:  # Service Unavailable
            print(f"Attempt {attempt + 1}/{max_retries}: Model is still loading. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
        else:
            print(f"Error: {response.status_code}, {response.text}")
            break
    else:
        print("Failed to get a response after multiple retries.")
        return None

# Query the API
output = query_with_wait(API_URL, headers, payload)

# Get current time for the filename
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filepath = rf'C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\result\Cross_Platform_Comparison\Llama3_3B_test\{current_time}.csv'

# Writing the output to a CSV file
if output:
    # Assuming the output is in JSON format, extract relevant information
    try:
        # Open the CSV file for writing
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write headers if necessary
            writer.writerow(["Timestamp", "API Response"])
            
            # Write the output into the CSV
            writer.writerow([current_time, json.dumps(output, indent=4)])
        
        print(f"Output successfully written to {filepath}")
    except Exception as e:
        print(f"Error writing output to CSV: {e}")
else:
    print("No output to write into the CSV file.")