import json
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnableSequence


load_local_model = True
if load_local_model:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    # Define the local model path
    local_model_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Llama-3.2-1B"

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
        #model_id="gpt2",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 50},
        device=0  # Use GPU
    )


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

### Output:
{answer}
"""
prompt_template = PromptTemplate(
    template=template_text,
    input_variables=["correct_answers", "examples_text", "test", "answer"],
)


# Load JSON data for correct answers
with open(r'Lambda_Feedback_Gao\functions\python_script\structured_prompts\LongChain\correct_answers.json', "r") as file:
    correct_answers_temp = json.load(file)
with open(r"Lambda_Feedback_Gao\functions\python_script\structured_prompts\LongChain\example_text.json", "r") as file:
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
test = "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. KFC takeaway, 2. Energy usage monitoring, 3. Smart parking systems."
answer = None

prompt_template = PromptTemplate(
    template=template_text,
    input_variables=["correct_answers", "examples_text", "test", "answer "],
)

# Build the full prompt
full_prompt = prompt_template.format(
    correct_answers=correct_answers,
    examples_text=examples_data,
    test= test,
    answer = answer
)
print(full_prompt)
# Print the full prompt
#dprint(full_prompt)


# Define an output parser to extract binary response
parser = RegexParser(
    regex=r"\b(Correct|Incorrect)\b",  # Add a capturing group
    output_keys=["binary_response"]   # Match the capture group
)

# raw_output = hf(full_prompt)
# print("Raw Output:", raw_output)

# Create a chain as a sequence
chain = RunnableSequence(hf, parser)

# Call the chain with the full prompt
result = chain.invoke(full_prompt)

# Print the output response
print("Binary Response:", result["binary_response"])

