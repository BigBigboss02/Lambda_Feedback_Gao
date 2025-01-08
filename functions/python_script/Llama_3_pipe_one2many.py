from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

import torch
import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline

class Config:
    # Path to the .env file containing credentials
    # env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
    env_path = 'login_configs.env'
    load_dotenv(dotenv_path=env_path)

    def __init__(self):
        self.local_model_path = 'Llama-3.2-1B' # local llama not included in this repo
        self.load_local_model = False
        # self.env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
        self.openai_url = os.getenv("OPENAI_URL")
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        # #repeative testing related
        # self.examples_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\functions\python_script\structured_prompts\examples1.json"
        # self.csv_saving_basepath = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\test_results\confusion_matrix"
        # self.repetive_test_num = 5

config = Config()

# Set default device
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the model and tokenizer
model_id = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Handle missing padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)

# Create the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
hf = HuggingFacePipeline(pipeline=pipe)

# Define the prompt template
template_text = """
### Instruction:
You are checking if the input is included in the following list, if yes the response will be true, else not:
{list}

### Input:
{word}

Response:
"""
prompt_template = PromptTemplate(
    template=template_text,
    input_variables=["list", "word"]
)

correct_answers = 'Chinese; Math; Russian; Physics; Chemistry; Biology; Communism; Geography; History '
input_word = 'History'

chain = prompt_template | hf.bind(skip_prompt=True)

# Invoke the chain
result = chain.invoke({
    "list": correct_answers,
    "word": input_word
})

# Print the result
print(result)
