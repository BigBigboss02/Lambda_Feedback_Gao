import json
import os
import re
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
from langchain.schema.runnable import RunnableSequence
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.schema import BaseOutputParser
import requests

# Configuration Class
class Config:
    def __init__(self):
        self.env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
        load_dotenv(dotenv_path=self.env_path)
        self.local_model_path = r"Llama-3.2-1B"
        self.load_local_model = True
        self.openai_url = os.getenv("OPENAI_URL")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize configuration
config = Config()

# Hugging Face Model Setup
def setup_hf_pipeline(local_model_path, use_local_model=True):
    if use_local_model:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForCausalLM.from_pretrained(local_model_path)
        hf_pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            device=0,  # GPU
            max_new_tokens=50
        )
        return HuggingFacePipeline(pipeline=hf_pipeline)
    else:
        return HuggingFacePipeline.from_model_id(
            model_id="gpt2",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 50}
        )

hf = setup_hf_pipeline(config.local_model_path, config.load_local_model)

# GPT-4-O Mini Agent Class
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
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
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

# Prompt Template and Input
def load_prompt_data():
    with open(r'Lambda_Feedback_Gao\functions\python_script\structured_prompts\LongChain\correct_answers_shorten.json', "r") as file:
        correct_answers_temp = json.load(file)
    with open(r"Lambda_Feedback_Gao\functions\python_script\structured_prompts\LongChain\example_text_shorten.json", "r") as file:
        examples_temp = json.load(file)

    correct_answers = "\n".join(correct_answers_temp["example_text"])
    examples_data = "\n".join(
        [
            f"Input: {example['original']}\nOutput: {'Correct' if example['correct'] else 'Incorrect'}"
            for example in examples_temp
        ]
    )
    return correct_answers, examples_data

correct_answers, examples_data = load_prompt_data()

test_input = '''
Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. KFC takeaway, 2. Energy usage monitoring, 3. Smart parking systems. 
Output:
'''

prompt_template = PromptTemplate(
    template="""
    Below is an instruction to determine right or wrong on student's coursework answers, paired with an input that provides further context. Give binary response that appropriately completes the request.

    ### Instruction:
    You are checking if the input includes 3 valid Wireless Sensor Network (WSN) applications from the given list:
    {correct_answers}

    ### Examples:
    {examples_text}

    ### Input:
    {test}
    """,
    input_variables=["correct_answers", "examples_text", "test"]
)

full_prompt = prompt_template.format(
    correct_answers=correct_answers,
    examples_text=examples_data,
    test=test_input,
)

# Combined Parser
def combined_parser(text: str):
    last_output_match = re.search(r"Output:.*", text, re.DOTALL)
    if not last_output_match:
        raise ValueError("No 'Output:' found in the input text.")
    extracted_text = text[last_output_match.end():].strip()
    binary_match = re.search(r"\b(Correct|Incorrect)\b", extracted_text)
    return binary_match.group(1) if binary_match else f"extracted text: {extracted_text}"

# Runnable Chain
chain = RunnableSequence(
    gpt4o_agent,
    combined_parser
)

# Test Harness Execution
if __name__ == "__main__":
    print("Full Prompt:")
    print(full_prompt)
    try:
        result = chain.invoke(full_prompt)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error during execution: {e}")
