from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
env_path = 'login_configs.env'
load_dotenv(dotenv_path=env_path)
openai_api_key = os.getenv('OPENAI_API_KEY')


# Initialize the ChatOpenAI object with GPT-4
chat = ChatOpenAI(
    model="gpt-4o-mini",  # Use "gpt-4" or "gpt-4-turbo"
    temperature=0.7,  # Adjust for creativity
    max_tokens=2000,  # Limit on response tokens
    openai_api_key="your_openai_api_key"  # Replace with your API key
)

# Define messages for the conversation
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain how LangChain can integrate GPT-4.")
]

# Make a call to the model
response = chat(messages)

# Print the response
print("GPT-4 Response:", response.content)
