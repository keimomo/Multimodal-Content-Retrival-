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
            "Select the configured answer model:",
            choices=[
                f"{model[0]}" for model in model_name
            ]
        ).ask()
        self.chat_model = choice_model
        self.chat_provider = provider
       

    def set_chat_model(self, all_models: list[tuple[str, str]]):
        choice_model = questionary.select(
            "Select an answer provider/model (OpenAI is the supported answer path):",
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
            print(f"[model] Configured answer preference: {self.chat_provider}:{self.chat_model}")
        except ValueError as e:
            print(f"[credentials] No saved key for {model_provider}. Enter it at the hidden prompt to continue setup.")
            self.save_api_key(model_provider)
            self.chat_model = model_name
            self.chat_provider = model_provider
            print(f"[model] Configured answer preference: {self.chat_provider}:{self.chat_model}")
            return model_provider
        return None
    
    def set_translation_model(self, all_models: list[tuple[str, str]]):
        choice_model = questionary.select(
            "Select a transcription preference (the worker currently loads Whisper small):",
            choices=[
                f"{model[1]}:{model[0]}" for model in all_models
            ]
        ).ask()
        self.translation_model_name = choice_model.split(":")[1].split(" ")[1]
        print(f"[model] Transcription preference: Whisper {self.translation_model_name}. The worker currently loads small independently of this setting.")

    

    def set_api_key(self):
        provider = questionary.select(
            "Choose a provider for credential setup (OpenAI is the supported answer path):",
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
        key = getpass(f"{provider} API key (hidden; stored in the OS keyring): ").strip()

        if not key:
            raise ValueError("API key cannot be empty.")

        try:
            keyring.set_password(self.service, provider, key)
            print(f"[credentials] Saved {provider} API key to the OS keyring (value hidden).")
        except Exception as e:
            print(f"[error] Could not save the {provider} key in the OS keyring: {e}")


    def get_api_key(self, provider: str) -> str:
        key = keyring.get_password(self.service, provider)

        if not key:
            raise ValueError(f"No saved API key for {provider}.")

        return key