import sqlite3
import json

class SQLDB:
    def __init__(self, path="vrag.db"):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.create_table()
        self.initialize_default()
    
    def get_connection(self):
        connection = sqlite3.connect("vrag.db")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
    # called on every bootup to check if intial configuration is completed 
    # if not -> return a list of items needs to be completed inorder for the user to use this tool
    def setup_check(self, stage: str, status: str | None = None):
        if status:
            cursor = self.connection.execute(
                "SELECT key FROM settings WHERE stage = ? AND status = ?",
                (stage, status),
            )
        else:
            cursor = self.connection.execute(
                "SELECT key FROM settings WHERE stage = ?",
                (stage,),
            )
        return cursor.fetchall()

    def initialize_default(self):
        self.connection.execute("INSERT INTO settings (key, status, stage) VALUES ('workspace', 'Incomplete', 'initialization') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO settings (key, status, stage) VALUES ('llm_api_key', 'Incomplete', 'initialization') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO settings (key, status, stage) VALUES ('vector_db_api_key', 'Incomplete', 'initialization') ON CONFLICT DO NOTHING")

        self.connection.execute("INSERT INTO settings (key, stage) VALUES ('vector_db_connection', 'bootup') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO settings (key, stage) VALUES ('llm_client_connection', 'bootup') ON CONFLICT DO NOTHING")


        self.connection.execute("INSERT INTO api_keys (provider, status) VALUES ('openai', 'Incomplete') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO api_keys (provider, status) VALUES ('anthropic', 'Incomplete') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO api_keys (provider, status) VALUES ('google', 'Incomplete') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO api_keys (provider, status) VALUES ('pinecone', 'Incomplete') ON CONFLICT DO NOTHING")

        self.connection.execute("INSERT INTO models (model_name, model_provider, model_usage, is_default) VALUES ('gpt-4o-mini', 'openai', 'chat', TRUE) ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider, model_usage) VALUES ('gpt-4o', 'openai', 'chat') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider, model_usage) VALUES ('Claude 3.7 Sonnet', 'anthropic', 'chat') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider, model_usage) VALUES ('Claude 3.7 Haiku', 'anthropic', 'chat') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider, model_usage) VALUES ('Google Gemini 2.5 Flash', 'google', 'chat') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider, model_usage) VALUES ('Google Gemini 2.5 Pro', 'google', 'chat') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider, model_usage) VALUES ('Whisper base', 'openai', 'translation') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider, model_usage, is_default) VALUES ('Whisper small', 'openai', 'translation', TRUE) ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider, model_usage) VALUES ('Whisper medium', 'openai', 'translation') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider, model_usage) VALUES ('Whisper large', 'openai', 'translation') ON CONFLICT DO NOTHING")

    def update_setting(self, key: str, status: str):
        self.connection.execute("UPDATE settings SET status = ? WHERE key = ?", (status, key))
        self.connection.commit()

    def create_table(self):
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS workspaces (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id),
                path TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                error_message TEXT,
                transcript TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                source_id INTEGER NOT NULL REFERENCES sources(id),
                modality TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                status TEXT NOT NULL,
                text TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                status TEXT,
                stage TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS models (
                model_name TEXT PRIMARY KEY,
                model_provider TEXT NOT NULL REFERENCES api_keys(provider),
                model_usage TEXT NOT NULL,
                is_default BOOLEAN NOT NULL DEFAULT FALSE
            );

            CREATE TABLE  IF NOT EXISTS api_keys (
                provider TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """) 

        self.connection.commit()

    def health_check(self):
        try:
            self.connection.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False


# workspace related functions
    def check_workspace(self, name: str):
        cursor = self.connection.execute(
            "SELECT id FROM workspaces WHERE name = ?",
            (name,)
        )
        return cursor.fetchone()

    def get_all_workspaces(self):
        return self.connection.execute(
            "SELECT name FROM workspaces"
        ).fetchall()

    def get_data_by_workspace(self, name: str):
        workspace = self.check_workspace(name)
        if workspace is None:
            raise ValueError(f"Workspace {name} not found. Type /ws li to see all workspaces.")
        id = workspace[0]
        rows = self.connection.execute(
            "SELECT id, path, status FROM sources WHERE workspace_id = ?",
            (id,)
        ).fetchall()
        return [(id, path, status) for id, path, status in rows]

    def insert_workspace(self, name: str):
        if not name:
            raise ValueError("Workspace name cannot be empty.")

        # Commits on success; rolls back if an exception occurs.
        try:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO workspaces (name) VALUES (?) ON CONFLICT DO NOTHING",
                    (name,)
                )
        except sqlite3.Error as e:
            print(f"Error inserting workspace: {e}")
            return False
        return True

    def delete_workspace(self, name:str):
        if not name:
            raise ValueError("Workspace name cannot be empty.")
        
        try:
            with self.connection:
                rows = self.connection.execute(
                    "SELECT id, name from workspaces where name in (?, ?)",
                    ("uncategorized", name)
                ).fetchall()

                if len(rows) != 2: 
                    raise ValueError("Workspace not found. Exit and re-run the CLI.")
                
                ids_by_name = {workspace_name: workspace_id for workspace_id, workspace_name in rows}
                
                uncategorized_id = ids_by_name["uncategorized"]
                del_id = ids_by_name[name]

                self.connection.execute(
                    "UPDATE sources SET workspace_id = ? WHERE workspace_id = ?",
                    (uncategorized_id,del_id)
                )
                self.connection.execute(
                    "DELETE FROM workspaces WHERE id = ?",
                    (del_id,)
                )
        except sqlite3.Error as e:
            print(f"Error deleting workspace: {e}")
            return False
        return True

# data related functions
    def insert_data(self, workspace_name: str, path: str):
        try:
            workspace = self.check_workspace(workspace_name)
            if workspace is None:
                raise ValueError(f"Workspace {workspace_name} not found. Type /ws li to see all workspaces.")
            workspace_id = workspace[0]
            with self.connection:
                cursor = self.connection.execute(
                    "INSERT INTO sources (workspace_id, path, status) VALUES (?, ?, ?)",
                    (workspace_id, path, "pending"),
                )
            return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            if e.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE:
                print(f"Error: Data {path} is already registered.")
            else:
                print(f"Error inserting data: {e}")
            return None

    # Model related functions
    def get_chat_model(self, provider: str):
        if provider == "all":
            return self.connection.execute(
                "SELECT model_name, model_provider FROM models WHERE model_usage = 'chat' ORDER BY model_provider DESC"
            ).fetchall()
        else:
            return self.connection.execute(
                "SELECT model_name, model_provider FROM models WHERE model_usage = 'chat' AND model_provider = ?",
                (provider,)
            ).fetchall()

    def get_translation_model(self):
        return self.connection.execute(
            "SELECT model_name, model_provider FROM models WHERE model_usage = 'translation' ORDER BY model_provider DESC"
        ).fetchall()
    
    def update_default_model(self, model_name: str, model_provider: str):
        current_default = self.connection.execute(
            "SELECT model_name, model_provider FROM models WHERE is_default = TRUE"
        ).fetchone()
        if current_default:
            self.connection.execute("UPDATE models SET is_default = FALSE WHERE model_name = ? AND model_provider = ?", (current_default[0], current_default[1]))
        self.connection.execute("UPDATE models SET is_default = TRUE WHERE model_name = ? AND model_provider = ?", (model_name, model_provider))
        self.connection.commit()

    # API key related functions
    def check_api_key(self, provider: str):
        cursor = self.connection.execute(
            "SELECT status FROM api_keys WHERE provider = ?",
            (provider,)
        )
        return cursor.fetchone() 

    def update_api_key(self, provider: str, status: str):
        self.connection.execute("UPDATE api_keys SET status = ? WHERE provider = ?", (status, provider))
        self.connection.commit()
    
    def get_all_queued_data(self):
            return self.connection.execute(
                "SELECT id, path, status FROM sources WHERE status in (?,?, ?)", ('pending', 'transcribed', 'text_embeded',)
            ).fetchall()

    def get_transcription(self, connection, id: int):
        cursor = connection.execute(
            "SELECT transcript FROM sources WHERE id = ?", (id,)
        ).fetchone()
        if cursor is None:
            raise ValueError(f"Source with id {id} not found.")
        return cursor
    
    def get_transcription_by_chunk_id(self, chunk_id: int):
        cursor = self.connection.execute(
            "SELECT text FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if cursor is None:
            raise ValueError(f"Chunk with id {chunk_id} not found.")
        return cursor
        
    def update_source_status(self, connection, id: int, status: str, transcript: str = None,):
        if transcript is not None:
            connection.execute("UPDATE sources SET status = ?, transcript = ? WHERE id = ?", (status, json.dumps(transcript, ensure_ascii=False), id))
        else:
            connection.execute("UPDATE sources SET status = ? WHERE id = ?", (status, id))
        connection.commit()
    
    def insert_chunk(self, connection, source_id: int, modality: str, start_time: float, end_time: float, status: str, text: str):
        cursor = connection.execute(
            "INSERT INTO chunks (source_id, modality, start_time, end_time, status, text) VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, modality, start_time, end_time, status, text)
        )
        return cursor.lastrowid

    def close(self):
        self.connection.close()