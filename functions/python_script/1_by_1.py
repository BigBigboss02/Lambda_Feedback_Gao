import json
import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnableSequence

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
        self.csv_saving_basepath = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\test_results\confusion_matrix"
        self.repetive_test_num = 5




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
        #max_length=1028, #this 1028 defines the total token of input and output together
        device=0,  # Use GPU (set to -1 for CPU)
        max_new_tokens= 2,  # Limit the number of tokens generated
        min_new_tokens = 1
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


# Define the prompt template
template_list_elements = """
Below is an instruction to determine right or wrong on student's coursework answers, paired with an input that provides further context. Give binary response that appropriately completes the request.

### Instruction:
You are checking if the input includes 3 similar answers from the given list, and returning only True or False for if the Answer is Correct:
{correct_answers}

### Examples:
{examples_text}

### Input:
{test}
"""

template_list_elements_singular = """
### Instruction:
You are checking if the input is included in the examples, and returning:
- 'True' if the Input is included in the 
- 'False' if the answer is incorrect, 

### Examples:
{examples_text}

### Input:
{test}
"""

template_give_short_reason =  """
Below is an instruction to determine right or wrong on student's coursework answers, paired with an input that provides further context. Provide a binary response along with a short reason for your decision.

### Instruction:
You are checking if the input includes 3 similar answers from the given list, and returning:
- 'True' if the answer is correct, with a short reason why.
- 'False' if the answer is incorrect, with a short reason why.

### Examples:
{examples_question_1}
Binary Response: {examples_question_1}
Reason: {short_reason_question_1}

### Input:
{test}

### Output Format:
Binary Response: [True/False]
Reason: [Short reason explaining the response]

### Output:
"""

prompt_template_le = PromptTemplate(
    template=template_list_elements,
    input_variables=["correct_answers", "examples_text", "test"]
)

prompt_template_les = PromptTemplate(
    template=template_list_elements_singular,
    input_variables=["examples_text", "test"]
)

prompt_template_gsr = PromptTemplate(
    template=template_list_elements,
    input_variables=["examples_text", "test"]
)

chain = prompt_template | hf.bind(skip_prompt=True)

model_response = chain.invoke({
    'correct_answers':correct_answers,
    'examples_text':EXAMPLES,
    'test': format_inputs(current_input)
})
print(f'model response: {model_response}')








