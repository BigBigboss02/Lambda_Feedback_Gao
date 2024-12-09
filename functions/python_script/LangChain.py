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

config = Config()




if config.load_local_model:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    # Define the local model path
    local_model_path = config.local_model_path

    # Load the tokenizer and model from the local path
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_model_path)

    # Set up the Hugging Face pipeline
    hf_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
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



#Calling GPT4o-mini
from dotenv import load_dotenv
import requests
# Define GPT-4-O Mini Agent
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

# Initialize GPT-4-O Mini Agent
gpt4o_agent = GPT4OMiniAgent(
    api_url=config.openai_url,
    headers={"Authorization": f"Bearer {config.openai_api_key}"}
)


#Pipeline Template

import json
from langchain.prompts import PromptTemplate

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
    input_variables=["correct_answers", "examples_text", "test", "answer"],
)


# Load JSON data for correct answers
with open(r'Lambda_Feedback_Gao\functions\python_script\structured_prompts\LongChain\correct_answers_shorten.json', "r") as file:
    correct_answers_temp = json.load(file)
with open(r"Lambda_Feedback_Gao\functions\python_script\structured_prompts\LongChain\example_text_shorten.json", "r") as file:
    examples_temp = json.load(file)



if "example_text" in correct_answers_temp and isinstance(correct_answers_temp["example_text"], list):
    correct_answers = "\n".join(correct_answers_temp["example_text"])  # Join list elements
else:
    raise ValueError("Unexpected structure in 'correct_answers.json' for correct_answers")

if isinstance(examples_temp, list):
    examples_data = "\n".join(
        [
            f"Input: {example['original']}\nOutput: {'Correct' if example['correct'] else 'Incorrect'}"
            for example in examples_temp
        ]
    )
else:
    raise ValueError("Unexpected structure in 'example_text.json' for examples")



# Define the test input
test = '''
Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. KFC takeaway, 2. Energy usage monitoring, 3. Smart parking systems. 
Output:
       '''



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
print(full_prompt)
# Print the full prompt
#dprint(full_prompt)





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







# #Calling the chain
chain = RunnableSequence(
    hf,
    combined_parser   # Combined parser pipeline
)

# Call the chain with the full prompt
result = chain.invoke(full_prompt)

print(f'binary result: {result}')