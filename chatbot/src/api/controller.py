from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src import sqlite_conn
from src.api import repository
from src.api.models import ChatUpdate, InvokeChatbotRequestBody, InvokeChatbotResponse
from src.core.chatbot import build_chatbot, MemoryType

chatbot = build_chatbot(MemoryType.SQLITE)
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/chats")
async def get_chats():
    return repository.get_chats()


@app.get("/chats/{chat_id}")
async def get_chat_by_id(chat_id: str):
    chat = repository.get_chat_by_id(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.put("/chats/{chat_id}")
async def update_chat(chat_id: str, chat_body: ChatUpdate):
    chat = repository.get_chat_by_id(chat_id)
    print(chat)
    if chat is None:
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


@app.post("/chatbot/invoke")
async def invoke_chatbot(body: InvokeChatbotRequestBody):
    """
    Invoke the chatbot with the given input.
    Use optional field chat_id to specify the chat history to use.
    If chat_id is not specified, create a new chat in the database table "chats".
    """
    chat_id = body.chat_id
    user_input = body.user_input

    if user_input is None or user_input.strip() == "":
        raise HTTPException(status_code=400, detail="User input is required.")

    # If chat_id is not provided, create a new chat
    if not chat_id:
        chat_id = repository.create_chat("Test Chat")
    else:
        chat = repository.get_chat_by_id(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Provided invalid chat_id.")

    # Prepare inputs for the chatbot
    inputs = {
        "messages": [HumanMessage(content=user_input)],
        "query": None,
        "relevance": None,
        "answer": None,
        "top_3": [],
        "relevant_part_texts": [],
        "valid_rag_answer": None
    }

    config = {"configurable": {"thread_id": chat_id}}
    chatbot_response = chatbot.invoke(inputs, config)

    return InvokeChatbotResponse(
        chat_id=chat_id,
        user_input=user_input,
        answer=chatbot_response["answer"],
        relevant_part_texts=chatbot_response.get("relevant_part_texts", [])
    )
