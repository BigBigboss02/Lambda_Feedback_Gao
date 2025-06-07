import os
from dotenv import load_dotenv

class Config:
    # Path to the .env file containing credentials
    env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\login_configs.env"
    load_dotenv(dotenv_path=env_path)

    def __init__(self, model='gpt', endpoint='api'):
        """
        Initialize the configuration with the specified model and endpoint.
        :param model: 'llama' or 'gpt'
        :param endpoint: 'api' or 'local'
        """
        self.model = model  # 'llama' or 'gpt'
        self.endpoint = endpoint  # 'api' or 'local'

        # Validate the model
        if self.model not in ['llama', 'gpt']:
            raise ValueError(f"Unknown model: {self.model}")

        # Set URLs and keys dynamically
        if self.endpoint == 'api':
            self.API_URL, self.AUTHORIZATION = self._get_model_config()
        elif self.endpoint == 'local':
            self.MODEL_PATH = self._get_model_config()
        # Set examples path dynamically
        current_dir = os.path.dirname(__file__)
        self.examples_path = os.path.join(current_dir, "structured_prompts", "examples1.json")

    def _get_model_config(self):
        """
        Determine API URL and authorization keys based on the model.
        :return: (API_URL, AUTHORIZATION) tuple
        """
        if self.model == 'llama':
            if self.endpoint == 'api':
                return (
                    os.getenv("3B_API_URL"),
                    os.getenv("HUGGINGFACE_AUTHORIZATION")
                )
            else:
                # modify the path where to load model here
                return (r'C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Llama-3.2-1B')
        elif self.model == 'gpt':
            if self.endpoint == 'api':
                return (
                    os.getenv("OPENAI_URL"),
                    os.getenv("OPENAI_API_KEY")
                )
            else:
                #OpenAi is not available for offline calling
                return (None)
        
config_gpt = Config(model='gpt')

