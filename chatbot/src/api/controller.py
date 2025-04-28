from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite import SqliteSaver

from src import sqlite_conn
from src.api import repository
from src.api.models import ChatUpdate

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/chats")
async def get_chats():
    return repository.get_chats()


@app.get("/chats/{chat_id}")
async def get_chat_by_id(chat_id: str):
    return repository.get_chat_by_id(chat_id)


@app.put("/chats/{chat_id}")
async def update_chat(chat_id: str, chat_body: ChatUpdate):
    chat = repository.get_chat_by_id(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    updates = {k: v for k, v in chat_body.model_dump().items() if v is not None}

    affected_rows = repository.update_chat(chat_id, updates)
    if affected_rows == 0:
        raise HTTPException(status_code=400, detail="Update failed.")

    return repository.get_chat_by_id(chat_id)


@app.get("/chat-history/{chat_id}")
async def get_chat_history(chat_id: str):
    state_by_chat_id = SqliteSaver(sqlite_conn).get({"configurable": {"thread_id": chat_id}})
    if state_by_chat_id is None:
        return []
    return state_by_chat_id.get("channel_values", {}).get("messages", [])


@app.delete("/chat-history/{chat_id}")
async def delete_chat_history_by_id(chat_id: str):
    if repository.delete_chat_history_by_id(chat_id):
        return JSONResponse(status_code=200, content={"message": "Deleted"})
    raise HTTPException(status_code=404, detail="Error deleting chat history")
