import sqlite3

from src.config import DB_DIR, DB_PATH

DB_DIR.mkdir(exist_ok=True)

sqlite_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
sqlite_conn.row_factory = sqlite3.Row
cursor = sqlite_conn.cursor()

try:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chats'")
    table_exists = cursor.fetchone()

    if table_exists:
        print("Chats table already exists.")
    else:
        cursor.execute("""
        CREATE TABLE chats (
            chat_id TEXT UNIQUE PRIMARY KEY,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        sqlite_conn.commit()
        print("Chats table CREATED.")

except sqlite3.Error as e:
    print(f"SQLite error occurred: {e}")

print("DB Initialized\n")
