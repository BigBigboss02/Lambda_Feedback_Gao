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
    input_variables=["correct_answers", "examples_text", "test"],
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
test = "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. KFC takeaway, 2. Energy usage monitoring, 3. Smart parking systems. Output:"


prompt_template = PromptTemplate(
    template=template_text,
    input_variables=["correct_answers", "examples_text", "test"],
)

# Build the full prompt
full_prompt = prompt_template.format(
    correct_answers=correct_answers,
    examples_text=examples_data,
    test=test
)

# Print the full prompt
print(full_prompt)