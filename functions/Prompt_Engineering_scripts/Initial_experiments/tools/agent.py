from dotenv import load_dotenv
import requests

# Define GPT-4-O Mini Agent
class GPT4OMiniAgent:
    def __init__(self, api_url, headers):
        self.api_url = api_url
        self.headers = headers

    def query(self, input_prompt: str):
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_prompt}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }

        # Send the query
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            # Extract assistant's output
            response_data = response.json()
            choices = response_data.get("choices", [])
            if choices:
                return choices[0]["message"]["content"]
            else:
                return "No valid choices returned from the model."
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")


# testing code as left below


# # Load environment variables
# env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
# load_dotenv(dotenv_path=env_path)

# api_url = os.getenv("OPENAI_URL")
# auth_key = os.getenv("OPENAI_API_KEY")
# headers = {"Authorization": f"Bearer {auth_key}"}

# # Initialize GPT-4-O Mini Agent
# gpt4o_agent = GPT4OMiniAgent(api_url=api_url, headers=headers)

# # Test input for the agent
# test_input = """
# Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. KFC takeaway, 2. Energy usage monitoring, 3. Smart parking systems.
# Is is correct or not?
# Response:
# """

# # Test the agent
# try:
#     print("Sending query to GPT-4-O Mini...")
#     result = gpt4o_agent.query(test_input)
#     print("Response from GPT-4-O Mini:")
#     print(result)
# except Exception as e:
#     print(f"Error during testing: {e}")
