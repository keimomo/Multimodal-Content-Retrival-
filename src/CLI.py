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
            print("[error] Local database check failed. Check access to vrag.db; startup cannot continue.")
            exit()
        else:
            print("[ready] Local database: vrag.db")
        check_list = self.sql_db.setup_check(stage='initialization', status='Incomplete')
        print(f"[setup] Remaining first-run steps: {len(check_list)}")
        if check_list:
            self.init_cli(check_list)
        
        self.bootup()
        self.start_workers()
        self.run()

    def init_cli(self, check_list):
        print("\nFirst-run setup\n---------------")
        for item in check_list:
            key = item[0]
            print(f"[setup] Configuring: {key}")
            if key == 'workspace':
                self.sql_db.insert_workspace("uncategorized")
                if self.sql_db.check_workspace("uncategorized"):
                    print("[ready] Default workspace: uncategorized")
                    self.sql_db.update_setting('workspace', 'Complete')
                else:
                    print("[error] Could not create the default workspace. Startup cannot continue.")
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
                    print(f"[error] Chat-provider setup failed: {e}")
                    exit()
            elif key == 'vector_db_api_key':
                try:
                    self.model_config.save_api_key("pinecone")
                    self.sql_db.update_api_key('pinecone', 'Complete')
                    self.sql_db.update_setting('vector_db_api_key', 'Complete')
                except Exception as e:
                    print(f"[error] Pinecone credential setup failed: {e}")
                    exit()

    """"
    Run bootup sequence for the CLI. (every bootup)
    -> Move to IDLE state to receive user input
    """
    def bootup(self):
        list_of_settings = self.sql_db.setup_check(stage='bootup')
        print(f"\n[startup] Preparing {len(list_of_settings)} services | embedding device: {self.device}")
        for setting in list_of_settings:
            key = setting[0]
            print(f"[startup] Step: {key}")
            if key == 'vector_db_connection':
                print("[connect] Pinecone index: mm-vectors-cli")
                try:
                    self.vector_db = PineconeDatabase(get_api_key=self.model_config.get_api_key)
                    self.vector_db.connect()
                    self.sql_db.update_setting("vector_db_connection", "Complete")
                    print("[ready] Pinecone initialized: mm-vectors-cli.")
                except ValueError:
                    self.sql_db.update_setting("vector_db_api_key", "Incomplete")
                    print("[error] No saved Pinecone API key. Restart the CLI to enter one.")
                    raise SystemExit(1)
                except UnauthorizedError:
                    self.sql_db.update_setting("vector_db_api_key", "Incomplete")
                    print("[error] Pinecone rejected the saved API key. Restart the CLI to replace it.")
                    raise SystemExit(1)
                except sqlite3.Error as error:
                    print(f"[error] Could not save Pinecone connection status in vrag.db: {error}")
                    raise SystemExit(1)

            elif key == 'llm_client_connection':
                print("[startup] Preparing the answer service; the API client is initialized by cd chat.")
                try:
                    self.llm = LLM(get_api_key=self.model_config.get_api_key)
                    self.sql_db.update_setting('llm_client_connection', 'Complete')
                except ValueError:
                    self.sql_db.update_setting("llm_api_key", "Incomplete")
                    print("[error] No saved OpenAI API key. Restart the CLI to enter one.")
                    raise SystemExit(1)
                except Exception as e:
                    print(f"[error] Answer-service setup failed: {e}")
                    raise SystemExit(1)
            else:
                print(f"[warning] Unrecognized startup setting: {key}")
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
                print(f"[transcribe] Source #{id} | {Path(path).name}: transcript saved; queued for embedding.")
            time.sleep(1)
            

            while not self.embed_queue.empty() and not self.exit_event.is_set():
                if self.audio_processor.model is not None:
                    self.audio_processor.unload_model()
                if self.raw_process.embedding.model is None:
                    self.raw_process.embedding.load_model()

                id, path, status = self.embed_queue.get()
                self.handle_embed(id, path, status, worker_db)
                self.embed_queue.task_done()
                print(f"[worker] Source #{id} | {Path(path).name}: processing pass finished (input stage: {status}).")
            time.sleep(1)

    def run(self):
        print("\nVideo Moment Retrieval\n----------------------\nType help for commands. Use queue to view pending work.\n")
        self.state = "idle"
        while True:
            command = input(f"vrag | {self.current_workspace} | {self.state} > ").strip().lower().split(" ")
            if command[0] in self.commands:
                self.dispatch(command)
            else:
                print(f"[error] Unknown command: {' '.join(command)!r}. Type help (without a leading slash).")

    def quit(self, args=None):
        print("[shutdown] Stopping the worker. An active processing operation may take time to return...")
        self.raw_process.embedding.unload_model()
        self.sql_db.close()
        self.exit_event.set()
        self.process_thread.join()
        print("[shutdown] Worker stopped. Goodbye.")
        exit()
    
    def set_state(self, args):
        if len(args) == 0:
            print("[usage] cd <idle|data|chat>  -  Choose one mode.")
            return
        if len(args) > 1:
            print("[usage] cd <idle|data|chat>  -  Supply exactly one mode.")
            return
        if args[0] == "data":
            if self.raw_process.embedding.model is not None:
                self.raw_process.embedding.unload_model()
            self.state ="data"
            print("[mode] data | Background processing continues. Use data add <path> to queue a file.")
        elif args[0] == "idle":
            if self.raw_process.embedding.model is not None:
                self.raw_process.embedding.unload_model()

            self.state = "idle"
            print("[mode] idle | Background processing continues. Use queue to view work.")
        elif args[0] == "chat":
            self.llm.set_client(self.model_config.chat_provider)
            self.raw_process.embedding.load_model()
            self.state = "chat"
            print(f"[mode] chat | Configured answer model: {self.model_config.chat_provider}:{self.model_config.chat_model}")
            print("[hint] Ask with q: <question>. Retrieval searches all workspaces.")
        else:
            print(f"[error] Unknown mode: {args[0]!r}. Available modes: idle, data, chat.")
          


    def handle_help(self, args=None):
        print("""
VIDEO MOMENT RETRIEVAL - COMMAND REFERENCE
========================================
Enter commands without a leading slash. <...> marks a required argument.

WORKSPACES
  ws                    Show the selected workspace.
  ws ls                 List all workspaces.
  ws add <name>         Create a workspace.
  ws set <name>         Select a workspace for registration and file listings.
  ws data <name>        List sources and statuses in a named workspace.
  ws rm <name>          Remove a workspace; move its sources to uncategorized.
                        The uncategorized workspace cannot be removed.

FILES AND PROCESSING
  data add <path>       Register an existing MP4 and queue background processing.
  data ls               List sources and statuses in the selected workspace.
  queue                 List unfinished sources, including work in progress.

MODES AND QUESTIONS
  cd data               Enter data mode; background processing continues.
  cd idle               Enter idle mode; background processing continues.
  cd chat               Initialize the answer client and load query embeddings.
  q: <question>         Retrieve evidence and generate a timestamped answer.
                        Enter cd chat before querying.

MODEL PREFERENCES
  model                 Show configured model and provider preferences.
  model chat            Open the answer-provider/model selection menu.
  model translation     Select a transcription preference. The worker currently
                        loads Whisper small independently of this preference.

SESSION
  help                  Show this command reference.
  exit                  Request shutdown and wait for the worker to stop.

EXAMPLE SESSION
  ws add lectures
  ws set lectures
  data add ./data/raw/lecture.mp4
  queue
  data ls
  cd chat               (after processing has finished)
  q: what does the lecturer say about neural networks?
  exit

CURRENT BEHAVIOR
  - Input is lowercased. Use lowercase paths without spaces; quoted paths
    are not supported. Only lowercase .mp4 inputs are accepted by processing.
  - Processing stages: pending -> transcribed -> text_embeded -> completed.
    text_embeded is the stored spelling for the text-indexing stage.
  - Workspaces organize local records. Questions search the entire index.
  - Choose OpenAI for answers; other provider menus are not fully integrated.
  - Wait for ingestion to finish before cd chat; it shares the embedding model.
  - Model changes do not persist reliably across restarts. If you change the
    chat provider in a session, re-enter cd chat to rebuild the client.
""")
    
    def handle_workspace(self, args=None):
        if len(args) == 0:
            print(f"[workspace] Selected: {self.current_workspace}")
            return

        sub_commands = args[0]
        if sub_commands == "ls":
            list = self.sql_db.get_all_workspaces()
            print(f"\nWorkspaces ({len(list)})\n----------------")
            for (name, ) in list:
                print(f"  {name}")
        elif sub_commands == "add":
            if len(args) == 2:
                name = args[1].strip().lower()
                if self.sql_db.insert_workspace(name):
                    print(f"[workspace] Workspace available: {name}. Select it with ws set {name}.")
                else:
                    print(f"[error] Could not create workspace: {name}")
            else:
                print("[usage] ws add <name>  -  Supply one workspace name.")
        
        elif sub_commands == "rm":
            if len(args) == 2:
                name = args[1].strip().lower()
                if name == "uncategorized": 
                    print("[error] The default workspace uncategorized cannot be removed.")
                    return
                if self.sql_db.delete_workspace(name):
                    print(f"[workspace] Removed {name}; its sources were moved to uncategorized.")
                else:
                    print(f"[error] Could not remove workspace: {name}")
                if self.current_workspace == name:
                    self.current_workspace = "uncategorized"
                    print("[workspace] Selected: uncategorized")
            else:
                print("[usage] ws rm <name>  -  Supply one workspace name.")
            
        elif sub_commands == "data":
            name = args[1].strip().lower()
            try:
                rows = self.sql_db.get_data_by_workspace(name)
                print(f"\nSources in {name} ({len(rows)})\n  ID     Status         File / path")
                if len(rows) == 0:
                    return
                for (id, path, status) in rows:
                    print(f"  {id:<6} {status:<14} {Path(path).name}\n         Path: {path}")
            except ValueError as e:
                print(f"[error] Could not list workspace sources: {e}")
                return
            #show all data linked to given workspace
        elif sub_commands == "set":
            print(f"[workspace] Selecting: {args[1]}")
            name = args[1].strip().lower()
            if self.sql_db.check_workspace(name) is not None:
                self.current_workspace = name
                print(f"[workspace] Selected: {name}")
            else:
                print(f"[error] Workspace {name!r} does not exist. Use ws ls to list workspaces.")
        else:
            print(f"[error] Unknown workspace command: {sub_commands!r}. Use ws, ws ls, ws add, ws set, ws data, or ws rm.")
            return

    def handle_data(self, args=None):
        if len(args) == 0:
            print("[usage] data ls | data add <path>  -  Type help for examples.")
            return
        sub_commands = args[0]
        if sub_commands == "ls":
            list = self.sql_db.get_data_by_workspace(self.current_workspace)
            print(f"\nSources in {self.current_workspace} ({len(list)})\n  ID     Status         File / path")
            if len(list) == 0:
                return
            for (id, path, status) in list:
                print(f"  {id:<6} {status:<14} {Path(path).name}\n         Path: {path}")
        elif sub_commands == "add":
            if len(args) == 2:
                file_path = args[1].strip()
                absolute_path = Path(file_path).expanduser().resolve()
                if not absolute_path.exists():
                    print(f"[error] File not found: {absolute_path}. Check the path and filename case.")
                    return
                if not absolute_path.is_file():
                    print(f"[error] Expected a file, but this path is a directory or another item: {absolute_path}")
                    return
                id = self.sql_db.insert_data(self.current_workspace, str(absolute_path))
                if id is not None:
                    print(f"[queue] Registered source #{id} in {self.current_workspace}: {absolute_path}\n        Background processing is queued; use queue or data ls to inspect progress.")
                    self.audio_queue.put((id, absolute_path))
                else:
                    print(f"[error] Could not register {absolute_path.name}. See the database message above.")
        else:
            print(f"[error] Unknown data command: {sub_commands!r}. Use data ls or data add <path>.")
            return

    def handle_queue(self, args=None):
        rows = self.sql_db.get_all_queued_data()
        print(f"\nProcessing queue ({len(rows)} unfinished sources; includes active work)\n  ID     Status         File / path")
        if len(rows) == 0:
            return
        for (id, path, status) in rows:
            print(f"  {id:<6} {status:<14} {Path(path).name}\n         Path: {path}")
        return

    def handle_audio(self, id: int, path: str, worker_db: SQLDB):
        transcription = self.audio_processor.transcribe(path)
        self.sql_db.update_source_status(connection=worker_db, id=id, status="transcribed", transcript=transcription)
    
    def handle_embed(self, id: int, path: str, status: str, worker_db: SQLDB):
        self.raw_process.embedding.load_model()
        if status == "transcribed":
            print(f"[index:text] Source #{id} | {Path(path).name}: generating and uploading transcript embeddings...")
            self.raw_process.process_text(connection=worker_db, sql_db=self.sql_db, pinecone_db=self.vector_db, abs_path=path, data_id=id)
            self.embed_queue.put((id, path, "text_embeded"))
        elif status == "text_embeded":
            print(f"[index:video] Source #{id} | {Path(path).name}: segmenting and indexing video clips...")
            self.raw_process.process_video(connection=worker_db, sql_db=self.sql_db, pinecone_db=self.vector_db, abs_path=path, data_id=id)
        else:
            print(f"[error] Source #{id}: unsupported processing stage {status!r}.")
            return
    def handle_query(self, args=None):
        if len(args) == 0:
            print("[usage] q: <question>  -  First enter cd chat, then add your question after q:.")
            return
        user_query = " ".join(args).strip()
        print("\nQuestion\n--------")
        print(user_query)
        print("[search] Retrieving transcript and visual evidence across all workspaces...")
        hits = self.raw_process.semantic_search(self.vector_db, user_query)
        if len(hits) == 0:
            print("[search] No evidence passed the current retrieval threshold. Try another question or check indexed sources with data ls.")
            return
        print(f"[search] Selected {len(hits)} evidence item(s); preparing source text and frames...")
        for index, hit in enumerate(hits):
            if hit['modality'] == "video":
                frames = self.raw_process.extract_frames(hit['source_path'], interval=5.0, start=hit['start'], end=hit['end'])
                hits[index]['frames'] = frames
            else:
                row = self.sql_db.get_transcription_by_chunk_id(hit['chunk_id'])
                text = row[0]
                hits[index]['text'] = text
        print(f"[answer] Requesting a grounded answer from {self.model_config.chat_model}...")
        answer = self.llm.answer_question(user_query, hits, self.model_config.chat_model)
        self.display_answer(answer)
        print("\n[hint] Ask another question with q: <question>, or type help.\n")

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
                paragraphs.append(f"  Source: {source}\n  Evidence ({citation.evidence_type}): {citation.explanation}")

        if answer.limitations:
            paragraphs.append("Limitations\n-----------")
            paragraphs.extend(f"- {item}" for item in answer.limitations)
        
        print(f"\nAnswer | {answer.status.replace('_', ' ')}\n---------------------------\n" + "\n\n".join(paragraphs))
        



    def handle_model(self, args=None):
        if len(args) == 0:
            print(f"\nConfigured model preferences\n----------------------------\n  Answer:        {self.model_config.chat_provider}:{self.model_config.chat_model}")
            print(f"  Transcription: {self.model_config.translation_model} {self.model_config.translation_model_name} (preference; worker currently loads small)")
            print(f"  Embeddings:    {self.model_config.embedding_model}")
            print(f"  Vector store:  {self.model_config.vector_db_provider}\n  Use model chat or model translation to open a selection menu.")
            return
        sub_commands = args[0]
        if sub_commands == "chat":
            chat_model = self.sql_db.get_chat_model("all")
            if chat_model is None:
                print("[error] No chat models are available in the local model catalog.")
                return
            else:
                model_provider = self.model_config.set_chat_model(chat_model)
                if model_provider is None:
                    print("[model] Chat preference selected. Re-enter cd chat to initialize the selected provider client.")
                    return

                if self.sql_db.check_api_key(model_provider):
                    print(f"[credentials] Setup status recorded for {model_provider}. Re-enter cd chat before querying.")
                    self.sql_db.update_api_key(model_provider, 'Complete')
                else:
                    print(f"[error] No credential setup record found for {model_provider}. Use model chat to select/configure the provider.")
                    return
        elif sub_commands == "translation":
            translation_model = self.sql_db.get_translation_model()
            if translation_model is None:
                print("[error] No transcription models are available in the local model catalog.")
                return
            else:
                self.model_config.set_translation_model(translation_model)
        else:
            print(f"[error] Unknown model command: {sub_commands!r}. Use model, model chat, or model translation.")
            return
    def dispatch(self, command):
        handler = self.commands.get(command[0])
        args = command[1:]
        if handler is None:
            print(f"[error] Unknown command: {command!r}. Type help for available commands.")
            return
        try:
            handler(args)
        except Exception as e:
            print(f"[error] Command {' '.join(command)!r} could not finish: {e}")
            return

if __name__ == "__main__":
    cli = CLI()
