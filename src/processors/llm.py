from openai import OpenAI

class LLM:
    def __init__(self, get_api_key: callable):
        self.client = OpenAI(api_key=get_api_key("openai"))

    