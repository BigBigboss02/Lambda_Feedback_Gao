import csv
import re
import json
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer



def extract_outputs(data):
    """
    Extracts only the final 'Output' value corresponding to the actual input, ignoring examples or instructions.

    Args:
        data (str): The input text containing multiple entries with 'Output' fields.

    Returns:
        list: A list of final 'Output' values corresponding to actual inputs.
    """
    # Split data into sections based on iterations
    iterations = re.split(r"### Input:", data)
    outputs = []

    for section in iterations[1:]:  # Skip the first part as it doesn't contain valid iterations
        # Find the last 'Output:' in the section
        match = re.search(r"Output: (.+)", section)
        if match:
            # Add the matched output to the list
            outputs.append(match.group(1).strip())

    return outputs


# Specify your model path
model_name = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Llama-3.2-1B"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set a padding token if not already defined
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to accommodate new token

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Move model to the appropriate device (GPU or CPU)
model = model.to(device)

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
examples_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Llama-3.2-1B\promts\examples1.json"

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

# Tokenize the full prompt
inputs = tokenizer(
    full_prompt,
    return_tensors="pt",
    padding=True,  # Add padding
    truncation=True  # Ensure token count doesn't exceed max limit
)

# Move inputs to the same device as the model
inputs = {key: value.to(device) for key, value in inputs.items()}

# Filepath for the CSV file, named by the current system time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filepath = rf'C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\result\llama3_{current_time}.csv'

# Open the CSV file to record results
# Open the CSV file to record results
with open(filepath, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Input", "Response"])

    # Run the prompt 10 times and record the result
    for i in range(1, 11):
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            temperature=0.4,  # Balance creativity and determinism
            top_p=0.9,  # Encourage high-probability tokens
            num_return_sequences=1,  # Generate a single output
            max_new_tokens=50,  # Allow generation of 50 new tokens
            pad_token_id=tokenizer.pad_token_id  # Ensure proper padding
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the output
        trimmed_response = extract_outputs(response)

        # Print the iteration, input, and corresponding output
        print(f"Iteration {i}: Input: {test} | Response: {trimmed_response}")

        # Write the iteration, input, and corresponding output to the CSV
        writer.writerow([i, test, trimmed_response])

print(f"Results have been recorded to {filepath}")
