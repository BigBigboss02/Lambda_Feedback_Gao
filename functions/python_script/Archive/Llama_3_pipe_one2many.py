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
        self.llama8b_endpoint = os.getenv('LLAMA3_1_8B_DdEnd')

        # Define the prompt template
        self.listing_template_text = """
            ### Instruction:
            You are checking if the input is included in the following list, if yes the response will be true, else not:
            
            ### List:
            {list}

            ### Input:
            {word}

            Response:
            """
        # Define the prompt template
        self.reasoning_template_text = """
            ### Instruction:
            You are tasked with determining whether the student's short answer is reasonable and aligns with the given question. If the answer is correct or plausible, respond with 'True'; otherwise, respond with 'False'.

            ### Question:
            {question}

            ### Student's Answer:
            {answer}

            Response:
            """    

        self.answer_short_question_template="""
            You are a STEM course teacher determining whether the student's 'answer' to a 'question' is true or false. 
            Provide clear feedback based on the accuracy of the answer.

            Question: {question}

            Student's Answer: {student_answer}

            Your Evaluation:
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
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain.schema import HumanMessage
    endpoint = config.llama8b_endpoint
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Use "gpt-4" or "gpt-4-turbo"
        temperature=0.7,  # Adjust for creativity
        max_tokens=50,  # Limit on response tokens
        openai_api_key=config.openai_api_key # Replace with your API key
        # openai_api_base=openai_url
    )


    llm = HuggingFaceEndpoint(
        endpoint_url=f"{endpoint}",
        max_new_tokens=12,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.01,
        repetition_penalty=1.03,
    )
llm("What did foo say about bar?")



prompt_template = PromptTemplate(
    template=config.listing_template_text,
    input_variables=["list", "word"]
)

#check if in list
correct_answers = 'Chinese; Math; Russian; Physics; Chemistry; Biology; Communism; Geography; History '
input_word = 'History'
#determing bolean result then give short reasoning
question = "Explain how the volume of a cube is calculated."
student_answer = "Multiply the length of one side by 4."

if config.mode == 'llama3_local':
    #chain = prompt_template | llm.bind(skip_prompt=True)
    chain = prompt_template | llm

    # Invoke the chain
    response = chain.invoke({
        "list": correct_answers,
        "word": input_word
    })
    print(response)

elif config.mode == 'gpt':
    gpt_prompt_template = PromptTemplate(template=config.answer_short_question_template,
        input_variables=["question", "student_answer"])
    chain = gpt_prompt_template | llm

    response = chain.invoke({
        "question": question,
        "student_answer": student_answer
    })

    print(response["content"])
