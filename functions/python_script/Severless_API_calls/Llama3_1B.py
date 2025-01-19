import requests

import os
from dotenv import load_dotenv
env_path = 'login_configs.env'
load_dotenv(dotenv_path=env_path)
HuggingAuth  = os.getenv("HUGGINGFACE_AUTHORIZATION")
llama3_2_url = os.getenv("LLAMA3_2_1B_URL")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HuggingAuth 
print(llama3_2_url)


#langchain calllin huggingface endpoints
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate






question = "Who won the FIFA World Cup in the year 1994? "

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)


repo_id = "meta-llama/Llama-3.2-1B"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    #headers=headers,
    max_length=5,
    temperature=0.2,
    huggingfacehub_api_token=HuggingAuth,
    server_kwargs={
        'x-wait-for-model': 'true',  # Wait for the model to be loaded
        'x-use-cache': 'false'       # Disable cache to test the same input multiple times
    }
)
llm_chain = prompt | llm
print(llm_chain.invoke({"question": question}))


# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
# headers = {"Authorization": "Bearer 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",
# 		   'x-wait-for-model': 'True', # we wanna wait for the model to be loaded
# 		   'x-use-cache': 'False'} # we wanna test the same input multiple times, so we don't want to use cache
# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# output = query({
# 	"inputs": "how many nucs do china have",
# 	'max_new_tokens': 10
# })

# print(output)