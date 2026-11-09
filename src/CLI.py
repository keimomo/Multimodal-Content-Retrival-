from database.sql_db import SQLDB
from config.model_config import ModelConfig
from database.vector_db import PineconeDatabase
from processors.audio_processor import AudioProcessor
from processors.raw_process import RawProcess
from processors.llm import LLM
from pinecone import UnauthorizedError, PineconeConnectionError, ApiError
import sqlite3
from pathlib import Path
from queue import Queue
from threading import Thread, Event
import time
import torch


class CLI:
    def __init__(self):
        self.sql_db = SQLDB()
        self.model_config = ModelConfig()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        self.audio_processor = AudioProcessor(self.device)
        

        self.vector_db = None
        self.llm = None
        self.raw_process = RawProcess(device=self.device)

        self.state = "idle"
        self.current_workspace = "uncategorized"
        self.commands = {
            "help": self.handle_help,
            "ws": self.handle_workspace,
            "data": self.handle_data,
            "cd": self.set_state,
            "model": self.handle_model,
            "exit": self.quit,
            "queue": self.handle_queue,
            "q:" : self.handle_query,
        }
        self.audio_queue = Queue()
        self.embed_queue = Queue()
        self.process_thread = Thread(target=self.process_worker, name = "process_worker")
        self.exit_event = Event()

        

        
        if not self.sql_db.health_check():
            print("Error: SQL database is not healthy. Please check the database connection. Exiting...")
            exit()
        else:
            print("SQL database is healthy.")
        check_list = self.sql_db.setup_check(stage='initialization', status='Incomplete')
        print(f"Found {len(check_list)} items to complete setup.")
        if check_list:
            self.init_cli(check_list)
        
        self.bootup()
        self.start_workers()
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
                    provider = self.model_config.set_api_key()
                    models = self.sql_db.get_chat_model(provider=provider)
                    self.model_config.set_model_provider(models, provider)

                    self.sql_db.update_api_key(provider, 'Complete')
                    self.sql_db.update_default_model(self.model_config.chat_model, self.model_config.chat_provider)
                    self.sql_db.update_setting('llm_api_key', 'Complete')
                except Exception as e:
                    print(f"Error: {e}")
                    exit()
            elif key == 'vector_db_api_key':
                try:
                    self.model_config.save_api_key("pinecone")
                    self.sql_db.update_api_key('pinecone', 'Complete')
                    self.sql_db.update_setting('vector_db_api_key', 'Complete')
                except Exception as e:
                    print(f"Error: {e}")
                    exit()

    """"
    Run bootup sequence for the CLI. (every bootup)
    -> Move to IDLE state to receive user input
    """
    def bootup(self):
        list_of_settings = self.sql_db.setup_check(stage='bootup')
        print(f"{len(list_of_settings)} items to before bootup.")
        for setting in list_of_settings:
            key = setting[0]
            print(f"Initiating {key} setup...")
            if key == 'vector_db_connection':
                print("Connecting to Pinecone...")
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
                try:
                    self.llm = LLM(get_api_key=self.model_config.get_api_key)
                    self.sql_db.update_setting('llm_client_connection', 'Complete')
                except ValueError:
                    self.sql_db.update_setting("llm_api_key", "Incomplete")
                    print("No OpenAI API key is saved. It will be requested on the next launch.")
                    raise SystemExit(1)
                except Exception as e:
                    print(f"Error: {e}")
                    raise SystemExit(1)
            else:
                print(f"Unknown setting: {key}")
    def start_workers(self):
        rows = self.sql_db.get_all_queued_data()
        for (id, path, status) in rows:
            if status == "pending":
                self.audio_queue.put((id, path))
            else:
                self.embed_queue.put((id, path, status))
        self.process_thread.start()
    
  


    def process_worker(self):
        worker_db = self.sql_db.get_connection()
        while not self.exit_event.is_set():
            while not self.audio_queue.empty() and not self.exit_event.is_set():
                
                if self.raw_process.embedding.model is not None:
                    self.raw_process.embedding.unload_model()
                if self.audio_processor.model is None:
                    self.audio_processor.load_model()

                id, path = self.audio_queue.get()
                self.handle_audio(id, path, worker_db)
                self.embed_queue.put((id, path, "transcribed"))
                self.audio_queue.task_done()
                print(f"Audio worker: {id, path} transcribed and added to embedding queue")
            time.sleep(1)
            

            while not self.embed_queue.empty() and not self.exit_event.is_set():
                if self.audio_processor.model is not None:
                    self.audio_processor.unload_model()
                if self.raw_process.embedding.model is None:
                    self.raw_process.embedding.load_model()

                id, path, status = self.embed_queue.get()
                self.handle_embed(id, path, status, worker_db)
                self.embed_queue.task_done()
                print(f"Embed worker: {id, path, status} processed")
            time.sleep(1)

    def run(self):
        print("Running CLI...")
        self.state = "idle"
        while True:
            command = input(f"vrag [{self.state} -> {self.current_workspace}]: ").strip().lower().split(" ")
            if command[0] in self.commands:
                self.dispatch(command)
            else:
                print(f"Unknown command: {' '.join(command)}. /help for available commands.")

    def quit(self, args=None):
        self.raw_process.embedding.unload_model()
        self.sql_db.close()
        self.exit_event.set()
        self.process_thread.join()
        print("Exiting...")
        exit()
    
    def set_state(self, args):
        if len(args) == 0:
            print(f"Too few arguments. Type /help for more information.")
            return
        if len(args) > 1:
            print(f"Too many arguments. Type /help for more information.")
            return
        if args[0] == "data":
            if self.raw_process.embedding.model is not None:
                self.raw_process.embedding.unload_model()
            self.state ="data"
        elif args[0] == "idle":
            if self.raw_process.embedding.model is not None:
                self.raw_process.embedding.unload_model()

            self.state = "idle"
        elif args[0] == "chat":
            self.llm.set_client(self.model_config.chat_provider)
            self.raw_process.embedding.load_model()
            self.state = "chat"
        else:
            print(f"Unknown state: {args[0]}")
          


    def handle_help(self, args=None):
        print("Available commands:")
    
    def handle_workspace(self, args=None):
        if len(args) == 0:
            print(f"Current workspace -> {self.current_workspace}")
            return

        sub_commands = args[0]
        if sub_commands == "ls":
            list = self.sql_db.get_all_workspaces()
            print(f"{len(list)} workspaces found:")
            for (name, ) in list:
                print(f"{name}")
        elif sub_commands == "add":
            if len(args) == 2:
                name = args[1].strip().lower()
                if self.sql_db.insert_workspace(name):
                    print(f"Workspace '{name}' added successfully.")
                else:
                    print(f"Error: Workspace '{name}' addition failed.")
            else:
                print("argument error. Type /help for more information.")
        
        elif sub_commands == "rm":
            if len(args) == 2:
                name = args[1].strip().lower()
                if name == "uncategorized": 
                    print("Error: Cannot remove 'uncategorized' workspace.")
                    return
                if self.sql_db.delete_workspace(name):
                    print(f"Workspace '{name}' removed successfully.")
                else:
                    print(f"Error: Workspace '{name}' removal failed.")
                if self.current_workspace == name:
                    self.current_workspace = "uncategorized"
                    print("Current workspace set to 'uncategorized'.")
            else:
                print("argument error. Type /help for more information.")
            
        elif sub_commands == "data":
            name = args[1].strip().lower()
            try:
                rows = self.sql_db.get_data_by_workspace(name)
                print(f"{len(rows)} data found:")
                if len(rows) == 0:
                    return
                for (id, path, status) in rows:
                    print(f"{id} - {Path(path).name} - {status}")
            except ValueError as e:
                print(f"Error: {e}. Type /help for more information.")
                return
            #show all data linked to given workspace
        elif sub_commands == "set":
            print("set the current workspace")
            name = args[1].strip().lower()
            if self.sql_db.check_workspace(name) is not None:
                self.current_workspace = name
                print(f"Current workspace set to '{name}'.")
            else:
                print(f"Error: Workspace '{name}' not found. Type /ws li to see all workspaces.")
        else:
            print(f"Unknown sub-command: {sub_commands}")
            return

    def handle_data(self, args=None):
        if len(args) == 0:
            print(f"Too few arguments. Type /help for more information.")
            return
        sub_commands = args[0]
        if sub_commands == "ls":
            list = self.sql_db.get_data_by_workspace(self.current_workspace)
            print(f"{len(list)} data found:")
            if len(list) == 0:
                return
            for (id, path, status) in list:
                print(f"{id} - {Path(path).name} - {status}")
        elif sub_commands == "add":
            if len(args) == 2:
                file_path = args[1].strip()
                absolute_path = Path(file_path).expanduser().resolve()
                if not absolute_path.exists():
                    print(f"Error: File {file_path} not found.")
                    return
                if not absolute_path.is_file():
                    print(f"Error: {file_path} is not a file.")
                    return
                id = self.sql_db.insert_data(self.current_workspace, str(absolute_path))
                if id is not None:
                    print(f"Data {absolute_path.name} added successfully to workspace: {self.current_workspace}.")
                    self.audio_queue.put((id, absolute_path))
                else:
                    print(f"Error: Data {absolute_path.name} addition failed.")
        else:
            print(f"Unknown sub-command: {sub_commands}")
            return

    def handle_queue(self, args=None):
        rows = self.sql_db.get_all_queued_data()
        print(f"{len(rows)} data found in the queue:")
        if len(rows) == 0:
            return
        for (id, path, status) in rows:
            print(f"{id} - {Path(path).name} - {status}")
        return

    def handle_audio(self, id: int, path: str, worker_db: SQLDB):
        transcription = self.audio_processor.transcribe(path)
        self.sql_db.update_source_status(connection=worker_db, id=id, status="transcribed", transcript=transcription)
    
    def handle_embed(self, id: int, path: str, status: str, worker_db: SQLDB):
        self.raw_process.embedding.load_model()
        if status == "transcribed":
            print(f"embedding transcribed text...")
            self.raw_process.process_text(connection=worker_db, sql_db=self.sql_db, pinecone_db=self.vector_db, abs_path=path, data_id=id)
            self.embed_queue.put((id, path, "text_embeded"))
        elif status == "text_embeded":
            print(f"embedding video...")
            self.raw_process.process_video(connection=worker_db, sql_db=self.sql_db, pinecone_db=self.vector_db, abs_path=path, data_id=id)
        else:
            print(f"Error: Unknown status: {status}")
            return
    def handle_query(self, args=None):
        if len(args) == 0:
            print(f"What would you like to query?")
            return
        user_query = " ".join(args).strip()
        print("--------------------------------")
        print("user: ", user_query)
        print("--------------------------------")
        hits = self.raw_process.semantic_search(self.vector_db, user_query)
        if len(hits) == 0:
            print("No hits found.")
            return
        for index, hit in enumerate(hits):
            if hit['modality'] == "video":
                frames = self.raw_process.extract_frames(hit['source_path'], hit['start'], hit['end'])
                hits[index]['frames'] = frames
            else:
                row = self.sql_db.get_transcription_by_chunk_id(hit['chunk_id'])
                text = row[0]
                hits[index]['text'] = text
        answer = self.llm.answer_question(user_query, hits, self.model_config.chat_model)
        self.display_answer(answer)
        print("--------------------------------")

    def timestamp(self, seconds):
        seconds = int(seconds)
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def display_answer(self, answer):
        paragraphs = []

        for block in answer.answer_blocks:
            paragraphs.append(block.text)

            for citation in block.citations:
                source = citation.source_id
                if (
                    citation.start_seconds is not None
                    and citation.end_seconds is not None
                ):
                    source += (
                        f" · {self.timestamp(citation.start_seconds)}"
                        f"–{self.timestamp(citation.end_seconds)}"
                    )
                paragraphs.append(f"*Source: {source}*")

        if answer.limitations:
            paragraphs.append("**Limitations**")
            paragraphs.extend(f"- {item}" for item in answer.limitations)
        
        print("\n\n".join(paragraphs))
        



    def handle_model(self, args=None):
        if len(args) == 0:
            print(f"Current Chat Model: {self.model_config.chat_model}")
            print(f"Current Translation Model: {self.model_config.translation_model} {self.model_config.translation_model_name}")
            print(f"Current Embedding Model: {self.model_config.embedding_model}")
            print(f"Current Vector Database: {self.model_config.vector_db_provider}")
            return
        sub_commands = args[0]
        if sub_commands == "chat":
            chat_model = self.sql_db.get_chat_model("all")
            if chat_model is None:
                print("No chat model found. Type /model chat to set a chat model.")
                return
            else:
                model_provider = self.model_config.set_chat_model(chat_model)
                if model_provider is None:
                    print("Chat model set successfully.")
                    return

                if self.sql_db.check_api_key(model_provider):
                    print(f"API key for {model_provider} set successfully.")
                    self.sql_db.update_api_key(model_provider, 'Complete')
                else:
                    print(f"API key for {model_provider} not found. Please set the API key.")
                    return
        elif sub_commands == "translation":
            translation_model = self.sql_db.get_translation_model()
            if translation_model is None:
                print("No translation model found. Type /model translation to set a translation model.")
                return
            else:
                self.model_config.set_translation_model(translation_model)
        else:
            print(f"Unknown sub-command: {sub_commands}")
            return
    def dispatch(self, command):
        handler = self.commands.get(command[0])
        args = command[1:]
        if handler is None:
            print(f"Unknown command: {command}. Type /help.")
            return
        try:
            handler(args)
        except Exception as e:
            print(f"Error: {e}")
            return

if __name__ == "__main__":
    cli = CLI() 