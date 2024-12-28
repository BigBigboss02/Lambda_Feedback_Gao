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
template_text = """
Below is an instruction to determine right or wrong on student's coursework answers, paired with an input that provides further context. Give binary response that appropriately completes the request.

### Instruction:
You are checking if the input includes 3 similar answers from the given list, and returning only True or False for if the Answer is Correct:
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
with open(r"Lambda_Feedback_Gao\functions\python_script\structured_prompts\confusion_matrix\A_Level_STEM_Inputs.json", "r") as file:
    inputs_temp = json.load(file)


examples_with_correctness = examples_temp["examples_with_correctness"]
example_text = correct_answers_temp["example_text"]
inputs= inputs_temp["examples_with_correctness"]
# Iterate through test examples and compare results
results = []


# Simulating the repeated execution of the loop 30 times
for _ in range(config.repetive_test_num):
    for subject, examples in examples_with_correctness.items():
        if subject in example_text:
            correct_answers = example_text[subject]  # Correct answers from example_text


        # # Define the list of examples
        # examples_list = [
        #     {'input': 'List 3 types of physics energy.', 'output': '1. Kinetic energy, 2. Potential energy, 3. Thermal energy.', 'correct': True},
        #     {'input': 'List 3 types of physics energy.', 'output': '1. Chemical energy, 2. Nuclear energy, 3. Elastic potential energy.', 'correct': True},
        #     {'input': 'List 3 types of physics energy.', 'output': '1. Kinetic energy, 2. Potential energy.', 'correct': False},
        #     {'input': 'List 3 types of physics energy.', 'output': '1. 12345, 2. 67890, 3. 24680.', 'correct': False}
        # ]

        # Template for formatting an example
        example_template = """
        Example:
        Input: {input}
        Answer: {answer}
        Correct: {correct}
        """

        # Build the prompt with four examples
        EXAMPLES = ""
        for example in examples[:4]:  # Select up to 4 examples
            EXAMPLES += example_template.format(
                input=example['input'],
                answer=example['output'],
                correct=example['correct']
            ).strip() + "\n\n"  # Add spacing between examples
        # print(EXAMPLES)
        # import time
        # time.sleep(100)

        if subject in inputs:
            for current_input in inputs[subject]:


                def format_inputs(example):
                    formatted_string = (
                        f"Input: {example['input']}\n"
                        f"Answer: {example['output']}\n"
                        f"Correct: "
                    )
                    return formatted_string

                # full_prompt = prompt_template.format(
                #     correct_answers=correct_answers,
                #     examples_text=EXAMPLES,
                #     test= format_inputs(current_input)
                # )
                # Define the chain

                chain = prompt_template | hf.bind(skip_prompt=True)
                # simple_test_input = (
                #     "Correct Answers: Mouth, Esophagus, Stomach\n\n"
                #     "Examples:\nExample Input: List 3 types of biology human digestive system.\n"
                #     "Example Answer: Mouth, Esophagus, Stomach.\nCorrect: True\n\n"
                #     "Test:\nInput: List 3 types of biology human digestive system.\n"
                #     "Answer: Mouth, Esophagus, Stomach.\nCorrect:"
                # )
                # print(hf(simple_test_input))
                # Invoke the chain
                model_response = chain.invoke({
                    'correct_answers':correct_answers,
                    'examples_text':EXAMPLES,
                    'test': format_inputs(current_input)
                })
                print(f'model response: {model_response}')

                # # Combine input_text into a single string for tokenization
                # input_text = (
                #     f"Correct Answers: {', '.join(correct_answers)}\n\n"
                #     f"Examples:\n{EXAMPLES}\n\n"
                #     f"Test:\n{format_inputs(current_input)}"
                # )

                # # Tokenize the combined string and calculate tokenized length
                # tokenized_length = len(tokenizer(input_text)['input_ids'])
                # print(f"Tokenized Length: {tokenized_length}")

                



                # # Compare with expected output
                # is_correct = (model_response == ("Correct" if current_input["correct"] else "Incorrect"))
                # results.append({
                #     "input": current_input["input"],
                #     "expected": "Correct" if current_input["correct"] else "Incorrect",
                #     "output": model_response,
                #     "result": "Pass" if is_correct else "Fail"
                # })
                # print(f'results: {results}')
                results.append({
                    "input": current_input["input"],
                    "expected": "Correct" if current_input["correct"] else "Incorrect",
                    "output": model_response,
                    "result": 'undefined'
                })

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




#Confusion Matrix Base CSV saving Area
import csv
from datetime import datetime

csv_saving_basepath = config.csv_saving_basepath 
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filepath = os.path.join(csv_saving_basepath, f"results_{current_time}.csv")
with open(filepath, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Input", "Expected Output", "Model Output", "Result"])  # Define column headers
with open(filepath, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for result in results:
        writer.writerow([result["input"], result["expected"], result["output"], result["result"]])

print(f"Results have been saved to {filepath}")





