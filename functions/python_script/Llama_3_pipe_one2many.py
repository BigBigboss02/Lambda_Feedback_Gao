from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json
from langchain_core.prompts import PromptTemplate

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
