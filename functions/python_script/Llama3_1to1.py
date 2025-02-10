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

    mode = 'gpt' #currently available option: gpt, llama3, llama3_local
    llama_version = '3_1_8B' # only available for llama3 mode

    debug_mode = False #Set to True to test the connection 
    temperature = 0.01
    max_new_token = 5
    cycle_num = 200
    skip_prompt = False #some models have their defalt prompt structure
    save_results = True
    if_plot = True
    template_type = 'templates_2D'  #options: 'templates_2D', 'templates_3D'
    dimension = '03'#options: '01', '02', '03','04'

    local_model_path = 'Llama-3.2-1B' # local llama not included in this repo
    example_path = '/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/semantic_comparisons_lower_pressure.csv'
    result_saving_path = 'test_results/1to1/cross_platform_experiments_1000trials/gpt4o_mini_00100050203'
    os.makedirs(result_saving_path, exist_ok=True) # Create the directory if it doesn't exist
    def __init__(self):
        self.openai_url = os.getenv("OPENAI_URL")
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.HuggingAuth  = os.getenv("HUGGINGFACE_AUTHORIZATION")
        self.llama3_2_repo_ID = os.getenv("LLAMA3_2_1B_REPO_ID")
        self.endpoint_3_1_8B = os.getenv("LLAMA3_1_8B_ENDPOINT")
        self.endpoint_3_2_1B = os.getenv("LLAMA3_2_1B_ENDPOINT")
        self.endpoint_3_2_3B = os.getenv("LLAMA3_2_3B_ENDPOINT")
        self.endpoint_3_3_70B = os.getenv("LLAMA3_3_70B_ENDPOINT")
        
        # Templates
        self.templates_2D = {
            "01": '''
            ### Examples:
            Word1: "happy", Word2: "happy"  
            Response: True

            Word1: "happy", Word2: "joyful"  
            Response: True

            Word1: "cat", Word 2: "dog"  
            Response: False

            Word1: "bank", Word 2: "actor"  
            Response: False

            ### Input:
            Word1:{target}, Word2:{word}
            Response:
            ''',
            "02": '''
            ### Instruction:
            Determine if the following two words are semantically similar. Provide one of the following responses:
            - "True" if the words are semantically the same.
            - "False" if the words are semantically different.

            ### Input:
            Word1:{target}, Word2:{word}

            ### Response:
            ''',
            "03": '''
            ### Instruction:
            Determine if the 2 words are semantically similar. Provide one of the following responses:
            - "True" if the words are semantically the same.
            - "False" if the words are semantically different.

            ### Examples:
            Word1: "happy", Word2: "happy"  
            Response: True

            Word1: "happy", Word2: "joyful"  
            Response: True

            Word1: "cat", Word 2: "dog"  
            Response: False

            Word1: "bank", Word 2: "actor"  
            Response: False

            ### Input:
            Word1:{target}, Word2:{word}

            ### Response:
            ''',
            
            "04": '''
            apple,malus,true
            pear,pear,true
            banana,apple,false
            red,blue,false
            color,colour,true
            box,boxes,true
            TV, computers, false
            
            {target},{word}, ...
            (complete the prompt with only True or False)
            '''
        }
        self.templates_3D = {
            "01": '''
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
            ''',
            "02": '''
            ### Instruction:
            Determine if the following two words are semantically similar. Provide one of the following responses:
            - "True" if the words are semantically the same.
            - "False" if the words are semantically different.
            - "Not Sure" if it is unclear based on the given words.

            ### Input:
            Word1:{target}, Word2:{word}

            ### Response:
            ''',
            "03": '''
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
            '''
        }

        self.prompt_template = getattr(self, self.template_type)[self.dimension]


config = Config()


# Set default device
if config.mode == 'llama3_local':
    # local models currently not available for testing, but can same code can be used to deploy modesls on 
    # universal gpu servers
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

if config.mode == 'gpt':
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Use "gpt-4" or "gpt-4-turbo"
        temperature=config.temperature,  # Adjust for creativity
        max_tokens=config.max_new_token,  # Limit on response tokens
        openai_api_key=config.openai_api_key # Replace with your API key
    )

elif config.mode == 'llama3':
    from langchain_huggingface import HuggingFaceEndpoint
    # from langchain.chains import LLMChain
    if config.llama_version == '3_1_8B':
        endpoint = config.endpoint_3_1_8B
    elif config.llama_version == '3_2_1B':
        endpoint = config.endpoint_3_2_1B
    elif config.llama_version == '3_2_3B':
        endpoint = config.endpoint_3_2_1B
    elif config.llama_version == '3_3_70B':
        endpoint = config.endpoint_3_3_70B


    llm = HuggingFaceEndpoint(
        endpoint_url=f"{endpoint}",
        # Specify the maximum input tokens (if supported by the model)
        # model_kwargs={"max_input_tokens": 4096},
        max_new_tokens=config.max_new_token,      
        temperature=config.temperature,
        huggingfacehub_api_token=config.HuggingAuth
    )
    
prompt_template = PromptTemplate(
    template=config.prompt_template,
    input_variables=["target", "word"]
)

#define a binary parser
import re
from langchain.schema.runnable import RunnableLambda

# Parser function to extract the last occurrence of "True" or "False"
def parse_last_boolean(response):
    matches = re.findall(r'\b(true|false)\b', response, re.IGNORECASE)
    return matches[-1].capitalize() if matches else "Unknown"

# Wrap parser in a LangChain Runnable
parser = RunnableLambda(parse_last_boolean)

if config.skip_prompt:
    chain = prompt_template | llm.bind(skip_prompt=True)
else:
    chain = prompt_template | llm | parser

if config.save_results:
    import pandas as pd
    from datetime import datetime
    results = []
    counter = 0
    examples_path = config.example_path
    df = pd.read_csv(examples_path)

    # ground truth validation
    for index, row in df.iterrows():
        if config.debug_mode:
            if counter >= 5:
                break
        else:
            if counter >= config.cycle_num:  # Ensure we don't exceed the loop limit
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
        if config.mode == 'gpt':
            content = response.content
        else:
            content = response
        # Print output for debugging
        print(counter)
        
        # Increment the counter
        counter += 1
        
        # Append the result as a tuple
        results.append((word1, word2, ground_truth, content))
    # Save results to a CSV file
    data = pd.DataFrame(results, columns=["Word1", "Word2", "Ground Truth", "Response"])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_saving_path = os.path.join(config.result_saving_path, f"{timestamp}.csv")
    data.to_csv(result_saving_path, index=False)
    print("Results saved")

    if config.if_plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        import os

        if config.template_type == 'templates_3D':
            label_mapping = {"True": 1, "False": 0, "Unsure": 2}
            data['Ground Truth'] = data['Ground Truth'].astype(str).str.strip().str.capitalize().map(label_mapping)
            data['Response'] = data['Response'].astype(str).str.strip().str.capitalize().map(label_mapping)

            # Filter only the necessary columns and clean the data
            filtered_data = data[['Ground Truth', 'Response']].dropna().astype(int)

            # Calculate the confusion matrix
            conf_matrix = confusion_matrix(filtered_data['Ground Truth'], filtered_data['Response'], labels=[1, 0, 2])

            # Plot the confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['True', 'False', 'Unsure'],
                        yticklabels=['True', 'False', 'Unsure'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()
        
        elif config.template_type == 'templates_2D':
            label_mapping = {"True": 1, "False": 0}
            data['Ground Truth'] = data['Ground Truth'].astype(str).str.strip().str.capitalize().map(label_mapping)
            data['Response'] = data['Response'].astype(str).str.strip().str.capitalize().map(label_mapping)


            # Filter only the necessary columns and clean the data
            filtered_data = data[['Ground Truth', 'Response']].dropna().astype(int)

            # Calculate the confusion matrix
            conf_matrix = confusion_matrix(filtered_data['Ground Truth'], filtered_data['Response'], labels=[1, 0])

            # Plot the confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['True', 'False'],
                        yticklabels=['True', 'False'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()

            # Save the plot in the same base folder as the CSV file
            plot_saving_path = os.path.join(config.result_saving_path, f"{timestamp}_confusion_matrix.png")
            plt.savefig(plot_saving_path, dpi=300)
            print(f"Plot saved at {plot_saving_path}")


            # plt.show()