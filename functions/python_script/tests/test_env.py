from dotenv import load_dotenv
import os

# Specify the path to the renamed .env file
env_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\environments\login_configs.env"
load_dotenv(dotenv_path=env_path)

class Config:
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_PORT = os.getenv("DB_PORT")
Config = Config()

# Test if variables are loaded
print("Database username:", Config.DB_USERNAME)
print("Database host:", Config.DB_HOST)
