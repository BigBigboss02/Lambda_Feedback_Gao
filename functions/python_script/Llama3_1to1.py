#1 set online access to the Llama3 portals
#2 push to 1 to 1 testing of llama3 structure
#3 finalise citations for paper writing 

#word llama
import os
import time
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


class Config:
    # Path to the .env file containing credentials
    # env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
    env_path = '/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/login_configs.env'
    load_dotenv(dotenv_path=env_path)
    mode = 'gpt' #currently available option: gpt, llama3_cloud, llama3_local

    def __init__(self):
        self.local_model_path = 'Llama-3.2-1B' # local llama not included in this repo
        self.load_local_model = False
        # self.env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
        self.openai_url = os.getenv("OPENAI_URL")
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.HuggingAuth  = os.getenv("HUGGINGFACE_AUTHORIZATION")
        self.llama3_2_repo_ID = os.getenv("LLAMA3_2_1B_REPO_ID")
        self.endpoint = os.getenv("LLAMA3_1_8B_DdEnd")
        self.INSTRUCTION = '''        
        ### Instruction:
        Determine if the following two words are semantically similar. Provide one of the following responses:
        - "True" if the words are semantically the same.
        - "False" if the words are semantically different.
        - "Not Sure" if it is unclear based on the given words.
        '''
        '''
        ### Examples:
        Word1: "happy", Word2: "happy"  
        Response: True

        Word1: "happy", Word2: "joyful"  
        Response: True

        Word1: "cat", Word 2: "dog"  
        Response: False

        Word1: "bank", Word 2: "actor"  
        Response: False

        Word1: "science", Word 2: "physics"  
        Response: unsure
        
        ### Input:
        Word1:{target}, Word2:{word}
        Response:
        '''
        # Define the prompt template
        self.semantic_comparison_template = """
        ### Instruction:
        Determine if the 2 words are semantically similar. Provide one of the following responses:
        - "True" if the words are semantically the same.
        - "False" if the words are semantically different.
        - "Unsure" if it is unclear based on the given words.

        ### Examples:
        Word1: "happy", Word2: "happy"  
        Response: True

        Word1: "happy", Word2: "joyful"  
        Response: True

        Word1: "cat", Word 2: "dog"  
        Response: False

        Word1: "bank", Word 2: "actor"  
        Response: False

        Word1: "science", Word 2: "physics"  
        Response: unsure
        
        ### Input:
        Word1:{target}, Word2:{word}
        
        ### Response:
        """ 
   
config = Config()


# Set default device
if config.mode == 'llama3_local':
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain.output_parsers import RegexParser
    from langchain_huggingface.llms import HuggingFacePipeline
    from langchain.schema.runnable import RunnableSequence
    import torch
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model and tokenizer
    model_id = config.local_model_path 
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Handle missing padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Create the pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    llm = HuggingFacePipeline(pipeline=pipe)

elif config.mode == 'gpt':
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Use "gpt-4" or "gpt-4-turbo"
        temperature=0.1,  # Adjust for creativity
        max_tokens=4,  # Limit on response tokens
        openai_api_key=config.openai_api_key # Replace with your API key
        # openai_api_base=openai_url
    )

elif config.mode == 'llama3_endpoint':
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain.chains import LLMChain

    repo_id = config.llama3_2_repo_ID 
    # llm = HuggingFaceEndpoint(
    #     repo_id=repo_id,
    #     max_new_tokens=5,
    #     temperature=0.2,
    #     huggingfacehub_api_token=config.HuggingAuth
    # )
    llm = HuggingFaceEndpoint(
        endpoint_url=f"{config.endpoint}",
        # Specify the maximum input tokens (if supported by the model)
        # model_kwargs={"max_input_tokens": 4096},
        max_new_tokens=5,      # Set a higher output token limit
        temperature=0.01,
        huggingfacehub_api_token=config.HuggingAuth
    )
    
prompt_template = PromptTemplate(
    template=config.semantic_comparison_template,
    input_variables=["target", "word"]
)
    

import pandas as pd


# chain = prompt_template | llm.bind(skip_prompt=True)
chain = prompt_template | llm


results = []
counter = 0
examples_path = '/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/semantic_comparisons.csv'
df = pd.read_csv(examples_path)

# ground truth validation
for index, row in df.iterrows():
    if counter >= 1000:  # Ensure we don't exceed the loop limit
        break

    # Simulate the word1 and word2 execution within the loop
    word1 = row["Word1"]
    word2 = row["Word2"]
    ground_truth = row["Ground Truth"]
    
    # Invoke the chain
    response = chain.invoke({
        "target": word1,
        "word": word2
    })
    content = response.content
    # Print output for debugging
    print(counter)
    
    # Increment the counter
    counter += 1
    
    # Append the result as a tuple
    results.append((word1, word2, ground_truth, content))
# Save results to a CSV file
data = pd.DataFrame(results, columns=["Word1", "Word2", "Ground Truth", "Response"])
data.to_csv('/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/week16_experiments/gpt_results_instNshot2.csv', index=False)

print("Results saved")

