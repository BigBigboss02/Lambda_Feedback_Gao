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
    env_path = 'login_configs.env'
    load_dotenv(dotenv_path=env_path)
    mode = 'llama3_endpoint' #currently available option: gpt, llama3_cloud, llama3_local

    def __init__(self):
        self.local_model_path = 'Llama-3.2-1B' # local llama not included in this repo
        self.load_local_model = False
        # self.env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
        self.openai_url = os.getenv("OPENAI_URL")
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.HuggingAuth  = os.getenv("HUGGINGFACE_AUTHORIZATION")
        self.llama3_2_repo_ID = os.getenv("LLAMA3_2_1B_REPO_ID")
       
        # Define the prompt template
        self.semantic_comparison_template = """
        ### Instruction:
        Determine if the following two words are semantically similar. Provide one of the following responses:
        - "True" if the words are semantically the same.
        - "False" if the words are semantically different.
        - "Not Sure" if it is unclear based on the given words.

        ### Examples:
        1. Word 1: "happy", Word 2: "joyful"  
        Response: True

        2. Word 1: "cat", Word 2: "dog"  
        Response: False

        3. Word 1: "bank", Word 2: "river"  
        Response: Not Sure

        ### Target:
        Word 1: {target}

        ### Input:
        Word 2: {word}

        Response:
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
        temperature=0.7,  # Adjust for creativity
        max_tokens=50,  # Limit on response tokens
        openai_api_key=config.openai_api_key # Replace with your API key
        # openai_api_base=openai_url
    )

elif config.mode == 'llama3_endpoint':
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain.chains import LLMChain

    repo_id = config.llama3_2_repo_ID 
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=5,
        temperature=0.2,
        huggingfacehub_api_token=config.HuggingAuth
    )
    prompt_template = PromptTemplate(
        template=config.semantic_comparison_template,
        input_variables=["target", "word"]
    )
    


if config.mode == 'llama3_endpoint':
    #chain = prompt_template | llm.bind(skip_prompt=True)
    chain = prompt_template | llm

    # Invoke the chain
    response = chain.invoke({
        "target": 'drinking',
        "word": 'drinking water'
    })
    print(response)