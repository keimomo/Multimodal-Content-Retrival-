import keyring
from getpass import getpass
import questionary

class ModelConfig:
    def __init__(self):
        #default settings
        self.chat_model = "gpt-4o-mini"
        self.chat_provider = "openai"

        self.embedding_model = "Embedding Gemma 2"

        self.translation_model = "Whis"

        self.vector_db_provider = "pinecone"
        self.service = "vrag-cli"

    

    def get_available_models(self, mode: str) -> list[str]:
        pass
    def set_model(self, mode: str, provider: str, model: str):
        pass
    

    def set_api_key(self):
        provider = questionary.select(
            "Select a provider to set the API key:",
            choices=[
                "openai",
                "anthropic",
                "google"
            ]
        ).ask()
        self.save_api_key(provider)


    def save_api_key(self, provider: str):
        # getpass hides what the user types.
        key = getpass(f"Enter your {provider} API key: ").strip()

        if not key:
            raise ValueError("API key cannot be empty.")

        try:
            keyring.set_password(self.service, provider, key)
            print(f"{provider} key saved.")
        except Exception as e:
            print(f"Error saving {provider} key: {e}")


    def get_api_key(self, provider: str) -> str:
        key = keyring.get_password(self.service, provider)

        if not key:
            raise ValueError(f"No saved API key for {provider}.")

        return key