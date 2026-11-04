from database.sql_db import SQLDB
from config.model_config import ModelConfig
from processors.embedding import MultiModalEmbedding
from database.vector_db import PineconeDatabase
from processors.llm import LLM
import keyring
from pinecone import UnauthorizedError, PineconeConnectionError, ApiError
import sqlite3

class CLI:
    def __init__(self):
        self.sql_db = SQLDB()
        self.model_config = ModelConfig()

        self.vector_db = None
        self.llm = None
        self.embedding_model = MultiModalEmbedding()

        
        if not self.sql_db.health_check():
            print("Error: SQL database is not healthy. Please check the database connection. Exiting...")
            exit()
        else:
            print("SQL database is healthy.")
        check_list = self.sql_db.setup_check(stage='initialization')
        print(f"Found {len(check_list)} items to complete setup.")
        if check_list:
            self.init_cli(check_list)
        
        self.bootup()
        self.run()

    def init_cli(self, check_list):
        print("Initializing CLI...")
        for item in check_list:
            key = item[0]
            print(f"Setting {key} is incomplete. Initiating setup...")
            if key == 'workspace':
                self.sql_db.insert_workspace("uncategorized")
                if self.sql_db.check_workspace("uncategorized"):
                    print("Workspace 'uncategorized' created successfully.")
                    self.sql_db.update_setting('workspace', 'Complete')
                else:
                    print("Error: Workspace 'uncategorized' creation failed. Exiting...")
                    exit()
            elif key == 'llm_api_key':
                try:
                    self.model_config.set_api_key()
                    self.sql_db.update_setting('llm_api_key', 'Complete')
                except Exception as e:
                    print(f"Error: {e}")
                    exit()
            elif key == 'vector_db_api_key':
                try:
                    self.model_config.save_api_key("pinecone")
                    self.sql_db.update_setting('vector_db_api_key', 'Complete')
                except Exception as e:
                    print(f"Error: {e}")
                    exit()

    """"
    Run bootup sequence for the CLI. (every bootup)
    -> Move to IDLE state to receive user input
    """
    def bootup(self):
        print("Running bootup sequence...")
        list_of_settings = self.sql_db.setup_check(stage='bootup')
        for setting in list_of_settings:
            key = setting[0]
            print(f"Initiating {key} setup...")
            if key == 'embedding_model_load':
                print("Loading embedding model...")
                self.embedding_model.load_model()
                self.sql_db.update_setting('embedding_model_load', 'Complete')
            elif key == 'vector_db_connection':
                try:
                    self.vector_db = PineconeDatabase(get_api_key=self.model_config.get_api_key)
                    self.vector_db.connect()
                    self.sql_db.update_setting("vector_db_connection", "Complete")
                except ValueError:
                    self.sql_db.update_setting("vector_db_api_key", "Incomplete")
                    print("No Pinecone API key is saved. It will be requested on the next launch.")
                    raise SystemExit(1)
                except UnauthorizedError:
                    self.sql_db.update_setting("vector_db_api_key", "Incomplete")
                    print("Pinecone rejected the saved API key. It will be requested again on the next launch.")
                    raise SystemExit(1)
                except sqlite3.Error as error:
                    print(f"Could not record the connection status: {error}")
                    raise SystemExit(1)

            elif key == 'llm_client_connection':
                print("Creating LLM client connection...")
                self.llm = LLM(get_api_key=self.model_config.get_api_key)
                self.sql_db.update_setting('llm_client_connection', 'Complete')
            else:
                print(f"Unknown setting: {key}")

    def run(self):
        print("Running CLI...")
        self.current_state = "idle"
        while True:
            command = input(f"vrag [{self.current_state}]: ")
            if command == "/exit":
                self.quit()
            else:
                print(f"Unknown command: {command}")

    def quit(self):
        self.embedding_model.unload_model()
        print("Exiting...")
        exit()

if __name__ == "__main__":
    cli = CLI() 