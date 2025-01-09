import os
import time
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


class Config:
    # Path to the .env file containing credentials
    # env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
    env_path = 'login_configs.env'
    load_dotenv(dotenv_path=env_path)
    mode = 'gpt' #currently available option: gpt, llama3_cloud, llama3_local

    def __init__(self):
        self.local_model_path = 'Llama-3.2-1B' # local llama not included in this repo
        self.load_local_model = False
        # self.env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
        self.openai_url = os.getenv("OPENAI_URL")
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Define the prompt template
        listing_template_text = """
            ### Instruction:
            You are checking if the input is included in the following list, if yes the response will be true, else not:
            
            ### List:
            {list}

            ### Input:
            {word}

            Response:
            """
        # Define the prompt template
        reasoning_template_text = """
            ### Instruction:
            You are tasked with determining whether the student's short answer is reasonable and aligns with the given question. If the answer is correct or plausible, respond with 'True'; otherwise, respond with 'False'.

            ### Question:
            {question}

            ### Student's Answer:
            {answer}

            Response:
            """    
        # #repeative testing related
        # self.examples_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\functions\python_script\structured_prompts\examples1.json"
        # self.csv_saving_basepath = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\test_results\confusion_matrix"
        # self.repetive_test_num = 5

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
    from langchain.chat_models import ChatOpenAI
    openai = OpenAI(
        model_name="gpt-3.5-turbo-instruct",
        http_client=httpx.Client(proxies="http://proxy.yourcompany.com:8080"),
    )



prompt_template = PromptTemplate(
    template=config.listing_template_text,
    input_variables=["list", "word"]
)

correct_answers = 'Chinese; Math; Russian; Physics; Chemistry; Biology; Communism; Geography; History '
input_word = 'History'

chain = prompt_template | llm.bind(skip_prompt=True)

# Invoke the chain
result = chain.invoke({
    "list": correct_answers,
    "word": input_word
})

# Print the result
print(result)
