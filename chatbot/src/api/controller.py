from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import TypeAdapter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware

from src.api import repository
from src.api.models import ChatUpdate, InvokeChatbotRequestBody, InvokeChatbotResponse, ChatHistoryEntry
from src.core.util import get_title_from_query
from src.db import init_db

origins = [
    "http://localhost:5173"
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=origins,  # Use ["*"] to allow all (dev only)
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    init_db()

    from src.core.chatbot import build_chatbot, MemoryType
    fastapi_app.state.chatbot = build_chatbot(MemoryType.SQLITE)

    yield


app = FastAPI(lifespan=lifespan, title="Chatbot API", middleware=middleware)


@app.get("/chats")
async def get_chats():
    try:
        return repository.get_chats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chats: {str(e)}")


@app.get("/chats/{chat_id}")
async def get_chat_by_id(chat_id: str):
    try:
        chat = repository.get_chat_by_id(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")
        return chat
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat: {str(e)}")


@app.put("/chats/{chat_id}")
async def update_chat(chat_id: str, chat_body: ChatUpdate):
    try:
        chat = repository.get_chat_by_id(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")

        updates = {k: v for k, v in chat_body.model_dump().items() if v is not None}
        affected_rows = repository.update_chat(chat_id, updates)
        if affected_rows == 0:
            raise HTTPException(status_code=400, detail="Update failed.")
        return repository.get_chat_by_id(chat_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating chat: {str(e)}")


@app.delete("/chats/{chat_id}")
async def delete_chat_by_id(chat_id: str):
    try:
        chat = repository.get_chat_by_id(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")

        repository.delete_chat_by_id(chat_id)
        return JSONResponse(status_code=200, content={"message": "Deleted"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chat: {str(e)}")


@app.get("/chat-history/{chat_id}")
async def get_chat_history(chat_id: str):
    try:
        chat_history_adapter = TypeAdapter(ChatHistoryEntry)
        chat_history = repository.get_chat_history_by_id(chat_id)

        # Group into (human, ai) pairs
        paired_history = []
        i = 0
        while i < len(chat_history) - 1:
            human = chat_history[i]
            ai = chat_history[i + 1]

            if isinstance(human, HumanMessage) and isinstance(ai, AIMessage):
                validated_human_entry = chat_history_adapter.validate_python({**human.__dict__, "chat_id": chat_id})
                validated_ai_entry = chat_history_adapter.validate_python({**ai.__dict__, "chat_id": chat_id})
                paired_history.append((validated_human_entry, validated_ai_entry))
                i += 2  # advance to next pair
            else:
                i += 1  # skip and check next

        return paired_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")


@app.delete("/chat-history/{chat_id}")
async def delete_chat_history_by_id(chat_id: str):
    try:
        repository.delete_chat_history_by_id(chat_id)
        return JSONResponse(status_code=200, content={"message": "Deleted"})
    except Exception:
        raise HTTPException(status_code=500, detail="Error deleting chat history")


@app.post("/chatbot/invoke")
async def invoke_chatbot(body: InvokeChatbotRequestBody):
    """
    Invoke the chatbot with the given input.
    Use optional field chat_id to specify the chat history to use.
    If chat_id is not specified, create a new chat in the database table "chats".
    """
    try:
        chat_id = body.chat_id
        user_input = body.user_input

        if user_input is None or user_input.strip() == "":
            raise HTTPException(status_code=400, detail="User input is required.")

        # If chat_id is not provided, create a new chat
        if not chat_id:
            chat_name = get_title_from_query(user_input)
            chat_id = repository.create_chat(chat_name)
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
        chatbot_response = app.state.chatbot.invoke(inputs, config)

        return InvokeChatbotResponse(
            chat_id=chat_id,
            user_input=user_input,
            answer=chatbot_response["answer"],
            relevant_part_texts=chatbot_response.get("relevant_part_texts", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking chatbot: {str(e)}")
