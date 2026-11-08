import keyring
from getpass import getpass
import questionary

class ModelConfig:
    def __init__(self):
        #default settings
        self.chat_model = "gpt-4o-mini"
        self.chat_provider = "openai"

        self.embedding_model = "Embedding Gemma 2"

        self.translation_model = "Whisper"
        self.translation_model_name = "small"

        self.vector_db_provider = "pinecone"
        self.service = "vrag-cli"

    
    def set_model_provider(self, model_name: list[str], provider: str):
        choice_model = questionary.select(
            "Select a model:",
            choices=[
                f"{model[0]}" for model in model_name
            ]
        ).ask()
        self.chat_model = choice_model
        self.chat_provider = provider
       

    def set_chat_model(self, all_models: list[tuple[str, str]]):
        choice_model = questionary.select(
            "Select a chat model:",
            choices=[
                f"{model[1]}:{model[0]}" for model in all_models
            ]
        ).ask()
        model_name = choice_model.split(":")[1]
        model_provider = choice_model.split(":")[0]
        try:
            self.get_api_key(model_provider)
            self.chat_model = model_name
            self.chat_provider = model_provider
            print(f"Chat model {self.chat_model} set successfully.")
        except ValueError as e:
            print(f"API key not found for {model_provider}. Please set the API key.")
            self.save_api_key(model_provider)
            self.chat_model = model_name
            self.chat_provider = model_provider
            print(f"Chat model {self.chat_model} set successfully.")
            return model_provider
        return None
    
    def set_translation_model(self, all_models: list[tuple[str, str]]):
        choice_model = questionary.select(
            "Select a translation model:",
            choices=[
                f"{model[1]}:{model[0]}" for model in all_models
            ]
        ).ask()
        self.translation_model_name = choice_model.split(":")[1].split(" ")[1]
        print(f"Translation model {self.translation_model_name} set successfully.")

    

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
        return provider


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