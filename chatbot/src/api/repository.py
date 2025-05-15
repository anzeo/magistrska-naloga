import sqlite3
import uuid
from sqlite3 import Error

from langgraph.checkpoint.sqlite import SqliteSaver

from src.db import cursor, sqlite_conn


def get_chats():
    cursor.execute("SELECT * FROM chats")
    results = cursor.fetchall()

    return results


def get_chat_by_id(chat_id):
    cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
    result = cursor.fetchone()

    return result


def create_chat(name):
    # Generate a unique chat ID and ensure it does not already exist in the database.
    chat_id = str(uuid.uuid4())
    # This should not happen, but in case it does, we generate a new, nonexistent ID
    while get_chat_by_id(chat_id) is not None:
        chat_id = str(uuid.uuid4())

    try:
        cursor.execute(
            "INSERT INTO chats (id, name) VALUES (?, ?)",
            (chat_id, name)
        )
        sqlite_conn.commit()

        return chat_id
    except Error as e:
        raise e


def update_chat(chat_id, updates):
    if updates == {}:
        return 0  # nothing to update

    try:
        # Build dynamic SET clause
        set_clause = ", ".join(f"{key} = ?" for key in updates.keys())
        values = list(updates.values())
        values.append(chat_id)

        sql = f"""
                UPDATE chats
                SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """
        cursor.execute(sql, values)
        sqlite_conn.commit()

        return cursor.rowcount
    except Error as e:
        raise e


def delete_chat_by_id(chat_id):
    try:
        cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        # Also delete chat history and don't commit yet
        delete_chat_history_by_id(chat_id, commit=False)
        # Commit the changes to the database after both deletions were successful
        sqlite_conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error during chat deletion: {e}")
        raise e


def delete_chat_history_by_id(chat_id, commit=True):
    try:
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (chat_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (chat_id,))
        if commit:
            sqlite_conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error during chat history deletion: {e}")
        raise e


def get_chat_history_by_id(chat_id):
    state_by_chat_id = SqliteSaver(sqlite_conn).get({"configurable": {"thread_id": chat_id}})
    if state_by_chat_id is None:
        return []
    return state_by_chat_id.get("channel_values", {}).get("messages", [])
