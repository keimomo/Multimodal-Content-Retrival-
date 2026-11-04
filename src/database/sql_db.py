import sqlite3


class SQLDB:
    def __init__(self, path="vrag.db"):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.create_table()
        self.initialize_default()
    

    # called on every bootup to check if intial configuration is completed 
    # if not -> return a list of items needs to be completed inorder for the user to use this tool
    def setup_check(self, stage: str):
         return self.connection.execute("SELECT KEY FROM settings WHERE STAGE = ? AND STATUS = 'Incomplete'", (stage,)).fetchall()


    def initialize_default(self):
        self.connection.execute("INSERT INTO settings (key, status, stage) VALUES ('workspace', 'Incomplete', 'initialization') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO settings (key, status, stage) VALUES ('llm_api_key', 'Incomplete', 'initialization') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO settings (key, status, stage) VALUES ('vector_db_api_key', 'Incomplete', 'initialization') ON CONFLICT DO NOTHING")

        self.connection.execute("INSERT INTO settings (key, status, stage) VALUES ('vector_db_connection', 'Incomplete', 'bootup') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO settings (key, status, stage) VALUES ('llm_client_connection', 'Incomplete', 'bootup') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO settings (key, status, stage) VALUES ('embedding_model_load', 'Incomplete', 'bootup') ON CONFLICT DO NOTHING")


        self.connection.execute("INSERT INTO api_keys (provider, status) VALUES ('openai', 'Incomplete') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO api_keys (provider, status) VALUES ('anthropic', 'Incomplete') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO api_keys (provider, status) VALUES ('google', 'Incomplete') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO api_keys (provider, status) VALUES ('pinecone', 'Incomplete') ON CONFLICT DO NOTHING")

        self.connection.execute("INSERT INTO models (model_name, model_provider) VALUES ('gpt-4o-mini', 'openai') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider) VALUES ('gpt-4o', 'openai') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider) VALUES ('Claude 3.7 Sonnet', 'anthropic') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider) VALUES ('Claude 3.7 Haiku', 'anthropic') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider) VALUES ('Google Gemini 2.5 Flash', 'google') ON CONFLICT DO NOTHING")
        self.connection.execute("INSERT INTO models (model_name, model_provider) VALUES ('Google Gemini 2.5 Pro', 'google') ON CONFLICT DO NOTHING")

    def update_setting(self, key: str, status: str):
        self.connection.execute("UPDATE settings SET status = ? WHERE key = ?", (status, key))
        self.connection.commit()

    def create_table(self):
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS workspaces (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                is_default BOOLEAN NOT NULL DEFAULT FALSE
            );

            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id),
                path TEXT NOT NULL,
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
                model_provider TEXT NOT NULL REFERENCES api_keys(provider)
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

    def check_workspace(self, name: str):
        cursor = self.connection.execute(
            "SELECT id FROM workspaces WHERE name = ?",
            (name,)
        )
        return cursor.fetchone() is not None

    def get_all_workspaces(self):
        return self.connection.execute(
            "SELECT name FROM workspaces"
        ).fetchall()

    def insert_workspace(self, name: str):
        name = name.strip()

        if not name:
            raise ValueError("Workspace name cannot be empty.")

        # Commits on success; rolls back if an exception occurs.
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO workspaces (name) VALUES (?) ON CONFLICT DO NOTHING",
                (name,)
            )

        return cursor.lastrowid

    def close(self):
        self.connection.close()