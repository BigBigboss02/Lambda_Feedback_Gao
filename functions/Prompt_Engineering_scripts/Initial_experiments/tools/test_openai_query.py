from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
env_path = 'login_configs.env'
load_dotenv(dotenv_path=env_path)
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_url = os.getenv("OPENAI_URL")

# Initialize the ChatOpenAI object with GPT-4


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Use "gpt-4" or "gpt-4-turbo"
    temperature=0.7,  # Adjust for creativity
    max_tokens=10,  # Limit on response tokens
    openai_api_key=openai_api_key # Replace with your API key
    # openai_api_base=openai_url
)

response = llm.invoke(messages)
print(response)