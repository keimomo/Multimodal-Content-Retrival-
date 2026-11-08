from collections.abc import Callable
from openai import OpenAI
from google import genai
from anthropic import Anthropic


class LLM:
    def __init__(self, get_api_key: Callable[[str], str]):
        self.get_api_key = get_api_key
        self.client = None

    def set_client(self, provider: str):
        if provider == "openai":
            self.client = OpenAI(api_key=self.get_api_key("openai"))
        elif provider == "anthropic":
            self.client = Anthropic(api_key=self.get_api_key("anthropic"))
        elif provider == "google":
            self.client = genai.Client(api_key=self.get_api_key("google"))
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    