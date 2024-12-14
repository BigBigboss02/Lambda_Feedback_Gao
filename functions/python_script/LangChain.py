import json
import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnableSequence

# from tools.agents import GPT4OMiniAgent

# Target Tasks: list objects, answer reason in 1 line, combining of both
# Input expected from teacher:
    #Type of the question, all needed objects and the core reason of the response
# Agents: 
    # Number checker for listing, feature extraction from reason agent as agent 1/1.5
    # Decision Maker as agent 2
        # Reason Checker
    # Feedback generator

class Config:
    # Path to the .env file containing credentials
    env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
    load_dotenv(dotenv_path=env_path)

    def __init__(self):
        self.local_model_path = r"Llama-3.2-1B" # 'llama' or 'gpt'
        self.load_local_model = True
        self.env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
        self.openai_url = os.getenv("OPENAI_URL")
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        #repeative testing related
        self.examples_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\functions\python_script\structured_prompts\examples1.json"
        self.csv_saving_basepath = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\test_results\confusion_matrix\confusion_matrix"
        self.repetive_test_num = 5
# Specify the path to the renamed .env file
env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
load_dotenv(dotenv_path=env_path)
  




config = Config()

if config.load_local_model:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    #init  GPU
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    
    # Define the local model path
    local_model_path = config.local_model_path

    # Setup Padding for Hf Tokeniser from Transformer
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(local_model_path)

    # Set up the Hugging Face pipeline
    hf_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_length=1028, #this 1028 defines the total token of input and output together
        device=0,  # Use GPU (set to -1 for CPU)
        max_new_tokens=50,  # Limit the number of tokens generated
    )

    hf = HuggingFacePipeline(pipeline=hf_pipeline)
else:
    # Initialize HuggingFacePipeline with GPU support
    hf = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 50}
    )



# Define GPT4-o mini Agent
from dotenv import load_dotenv
import requests
class GPT4OMiniAgent:
    def __init__(self, api_url, headers):
        self.api_url = api_url
        self.headers = headers

    def query(self, input_prompt: str):
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_prompt}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }

        # Send the query
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            # Extract assistant's output
            response_data = response.json()
            choices = response_data.get("choices", [])
            if choices:
                return choices[0]["message"]["content"]
            else:
                return "No valid choices returned from the model."
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
gpt4o_agent = GPT4OMiniAgent(
    api_url=config.openai_url,
    headers={"Authorization": f"Bearer {config.openai_api_key}"}
)


#Pipeline Template

import json
from langchain.prompts import PromptTemplate


def process_json_files(correct_answers_file, examples_file):
    """
    Process JSON files to extract and format "correct_answers" and "examples_text".

    Args:
        correct_answers_file (str): Path to the correct answers JSON file.
        examples_file (str): Path to the examples JSON file.

    Returns:
        tuple: A tuple containing:
            - correct_answers (str): Formatted correct answers.
            - examples_data (str): Formatted examples text.

    Raises:
        ValueError: If the JSON structure is unexpected.
    """
    # Load JSON data for correct answers
    with open(correct_answers_file, "r") as file:
        correct_answers_temp = json.load(file)

    with open(examples_file, "r") as file:
        examples_temp = json.load(file)

    # Process correct answers
    if "example_text" in correct_answers_temp and isinstance(correct_answers_temp["example_text"], list):
        correct_answers = "\n".join(correct_answers_temp["example_text"])
    else:
        raise ValueError("Unexpected structure in 'correct_answers.json' for correct_answers")

    # Process examples data
    if isinstance(examples_temp, list):
        examples_data = "\n".join(
            [
                f"Input: {example['original']}\nOutput: {'Correct' if example['correct'] else 'Incorrect'}"
                for example in examples_temp
            ]
        )
    else:
        raise ValueError("Unexpected structure in 'example_text.json' for examples")

    return correct_answers, examples_data

correct_answers_file = r'Lambda_Feedback_Gao\functions\python_script\structured_prompts\LongChain\correct_answers_shorten.json'
examples_file = r'Lambda_Feedback_Gao\functions\python_script\structured_prompts\LongChain\example_text_shorten.json'
correct_answers, examples_data = process_json_files(correct_answers_file, examples_file)
# Define the test input
test = '''
Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. KFC takeaway, 2. Energy usage monitoring, 3. Smart parking systems. 
Output:
       '''

# Define the prompt template
template_text = """
Below is an instruction to determine right or wrong on student's coursework answers, paired with an input that provides further context. Give binary response that appropriately completes the request.

### Instruction:
You are checking if the input includes 3 valid Wireless Sensor Network (WSN) applications from the given list:
{correct_answers}

### Examples:
{examples_text}

### Input:
{test}
"""
prompt_template = PromptTemplate(
    template=template_text,
    input_variables=["correct_answers", "examples_text", "test"],
)
# Build the full prompt
full_prompt = prompt_template.format(
    correct_answers=correct_answers,
    examples_text=examples_data,
    test= test,
)
print('---------------')
print(full_prompt)
print('---------------')
# Print the full prompt
#dprint(full_prompt)

import time
time.sleep(100)



# Parser
import re
from langchain.schema import BaseOutputParser
# Combine the parsing logic into a single callable function
def combined_parser(text: str):
    # Step 1: Extract the part after the last "Output:"
    last_output_match = re.search(r"Output:.*", text, re.DOTALL)
    if not last_output_match:
        raise ValueError("No 'Output:' found in the input text.")
    extracted_text = text[last_output_match.end():].strip()

    # Step 2: Use RegexParser to determine the binary response
    binary_match = re.search(r"\b(Correct|Incorrect)\b", extracted_text)
    if not binary_match:
        return(f"extracted text: {extracted_text}")
    return {"binary_response": binary_match.group(1)}







# # #Calling the chain
# chain =  full_prompt | hf #|combined_parser  


# # Call the chain with the full prompt
# result = chain.invoke()
# question = "What is electroencephalography?"

# print(chain.invoke({"question": question}))
