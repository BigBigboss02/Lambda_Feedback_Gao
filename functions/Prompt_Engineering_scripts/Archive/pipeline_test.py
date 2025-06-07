import os
import json
import requests
from transformers import pipeline as HuggingFacePipeline

# Configuration class to load environment variables
class Config:
    API_URL = os.getenv("OPENAI_URL")  # Replace with the correct key for GPT-4-O API URL
    AUTHORIZATION = os.getenv("OPENAI_API_KEY")  # Replace with the correct authorization key variable
    examples_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\functions\python_script\structured_prompts\examples1.json"

# Instantiate configuration
config = Config()

# Hugging Face pipeline configuration
hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 50}
)

# Prepare payload for GPT-4-O Mini API
def call_gpt4o_mini(full_prompt):
    api_url = config.API_URL
    headers = {"Authorization": f"Bearer {config.AUTHORIZATION}"}
    payload = {
        "model": "gpt-4o-mini",  # Specify GPT-4-O Mini as the model
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt}
        ],
        "max_tokens": 100,  # Limit response length
        "temperature": 0.7  # Control randomness
    }
    
    # Make the API call
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"API call failed with status {response.status_code}: {response.text}")

# Function to use Hugging Face pipeline
def call_huggingface_pipeline(prompt):
    result = hf(prompt)
    return result[0]["generated_text"]

# Integrated function
def process_prompt(full_prompt, use_gpt4o=True):
    """
    Processes the given prompt using GPT-4-O Mini API or Hugging Face pipeline.
    
    Args:
        full_prompt (str): The input prompt.
        use_gpt4o (bool): Whether to use GPT-4-O Mini API or Hugging Face pipeline.
    
    Returns:
        str: The response from the model.
    """
    try:
        if use_gpt4o:
            print("Using GPT-4-O Mini API...")
            return call_gpt4o_mini(full_prompt)
        else:
            print("Using Hugging Face pipeline...")
            return call_huggingface_pipeline(full_prompt)
    except Exception as e:
        print(f"Error during processing: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Load examples from JSON file
    examples_path = config.examples_path
    with open(examples_path, "r") as file:
        examples = json.load(file)

    # Format examples as part of the prompt
    examples_text = "\n".join(
        [f"Input: {item['original']}\nOutput: {'Correct' if item['correct'] else 'Incorrect'}"
         for item in examples]
    )

    # Build the full prompt
    prompt_list = [
        "Environmental monitoring",
        "Health care monitoring",
        "Industrial process control"
    ]
    prompt_text = "\n".join(prompt_list)
    full_prompt = f"""
Below is an instruction to determine right or wrong on student's coursework answers, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are checking if the input includes 3 valid Wireless Sensor Network (WSN) applications from the given list:
{prompt_text}

### Examples:
{examples_text}

### Input:
Give 3 examples of WSN applications. 1. Asset tracking, 2. Energy usage monitoring, 3. Smart parking systems.
"""

    # Use either GPT-4-O Mini API or Hugging Face pipeline
    result = process_prompt(full_prompt, use_gpt4o=True)  # Set use_gpt4o=False for Hugging Face pipeline
    print("Response:")
    print(result)
