import requests
import json

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-405B"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

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

# Path to the examples JSON file
examples_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\promts\examples1.json"

# Load examples from JSON file
with open(examples_path, "r") as file:
    examples = json.load(file)

# Format examples as part of the prompt
examples_text = "\n".join(
    [f"Input: {item['original']}\nOutput: {'Correct' if item['correct'] else 'Incorrect'}\n"
     for item in examples]
)
print(examples_text)

# Sample input data
# test = "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. Asset tracking, 2. Energy usage monitoring, 3. Smart parking systems."
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

output = query({
	"inputs": "Can you please let us know more details about your ",
})

print(output)