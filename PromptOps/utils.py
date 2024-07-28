import openai
import os

def set_openai_api_key(api_key=None):
    """
    Set the OpenAI API key.
    
    Parameters:
    api_key (str): The OpenAI API key. If not provided, it will be fetched from the environment variable OPENAI_API_KEY.
    """
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("OpenAI API key is not set. Please provide an API key.")