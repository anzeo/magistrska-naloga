import asyncio
import json
from collections import defaultdict
from contextlib import asynccontextmanager
from threading import Thread
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from pydantic import TypeAdapter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware

from src.api import repository
from src.api.models import ChatUpdate, InvokeChatbotRequestBody, InvokeChatbotStreamingResponse, ChatHistoryEntry, \
    ChatHistoryTurn
from src.api.util import format_sse, get_ai_act_part_by_id
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

    fastapi_app.state.chat_locks = defaultdict(lambda: asyncio.Lock())

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
async def get_chat_history(chat_id: str) -> List[ChatHistoryTurn]:
    try:
        chat_history_adapter = TypeAdapter(ChatHistoryEntry)
        chat_history = repository.get_chat_history_by_id(chat_id)

        human_messages = {}
        ai_messages = []

        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                human_messages[msg.id] = msg
            elif isinstance(msg, AIMessage):
                ai_messages.append(msg)

        paired_history = []
        for ai_msg in ai_messages:
            parent_id = ai_msg.additional_kwargs.get("parent_id")
            if parent_id and parent_id in human_messages:
                validated_human_entry = chat_history_adapter.validate_python(
                    {**human_messages[parent_id].__dict__, "chat_id": chat_id})
                validated_ai_entry = chat_history_adapter.validate_python({**ai_msg.__dict__, "chat_id": chat_id})

                if len(getattr(validated_ai_entry, "relevant_part_texts", [])) > 0:
                    # If there are relevant parts, fetch their full content
                    for part in validated_ai_entry.relevant_part_texts:
                        part_id = part.get("id", None)
                        part["full_content"] = get_ai_act_part_by_id(part_id)

                paired_history.append(ChatHistoryTurn(human=validated_human_entry, ai=validated_ai_entry))

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
            existing_chat = repository.get_chat_by_id(chat_id)
            if not existing_chat:
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

        async def event_stream():
            # Allow concurrent access to the same chat_id
            async with app.state.chat_locks[chat_id]:
                try:
                    chat_data = repository.get_chat_by_id(chat_id)
                    yield format_sse(json.dumps({"type": "chat_data", "v": dict(chat_data)}), event="message")

                    chatbot_response = None

                    answer_to_stream = []

                    queue = asyncio.Queue()

                    # Function to run the streaming in a separate thread, so the chatbot invoke does not block
                    def run_streaming():
                        try:
                            for mode, chunk in app.state.chatbot.stream(inputs, config,
                                                                        stream_mode=["custom", "values", "messages"]):
                                queue.put_nowait((mode, chunk))
                        except Exception as e:
                            queue.put_nowait(("error", str(e)))
                        finally:
                            queue.put_nowait(("done", None))

                    # Start stream in background
                    thread = Thread(target=run_streaming)
                    thread.start()

                    while True:
                        mode, chunk = await queue.get()
                        if mode == "done":
                            break
                        elif mode == "error":
                            yield format_sse(json.dumps({"error": chunk}), event="error")
                        elif mode == "custom":
                            yield format_sse(json.dumps({"type": "step", "v": chunk}), event="message")
                        elif mode == "values":
                            # Store response for later use
                            chatbot_response = chunk
                        elif mode == "messages":
                            msg, metadata = chunk
                            # if metadata["langgraph_node"] == "RAG_Answer":
                            #     if isinstance(msg, AIMessageChunk):
                            #         answer_to_stream.append(msg.content)
                            #     print(metadata)
                            #     print(msg)
                            if metadata["langgraph_node"] in ["Invalid_RAG_Answer", "LLM"]:
                                if isinstance(msg, AIMessageChunk):
                                    answer_to_stream.append(msg.content)

                    # Then stream the answer
                    for chunk in answer_to_stream if len(answer_to_stream) else chatbot_response["answer"]:
                        yield format_sse(json.dumps({"v": chunk}), event="answer")
                        await asyncio.sleep(0.02)

                    chat_history_adapter = TypeAdapter(ChatHistoryEntry)

                    human_msg, ai_msg = chatbot_response["messages"][-2:]

                    validated_human_entry = chat_history_adapter.validate_python(
                        {**human_msg.__dict__, "chat_id": chat_id})
                    validated_ai_entry = chat_history_adapter.validate_python({**ai_msg.__dict__, "chat_id": chat_id})

                    if len(getattr(validated_ai_entry, "relevant_part_texts", [])) > 0:
                        validated_ai_entry.relevant_part_texts = [
                            {**part.__dict__, "full_content": get_ai_act_part_by_id(getattr(part, "id", None))}
                            for part in validated_ai_entry.relevant_part_texts
                        ]

                    final_response = InvokeChatbotStreamingResponse(
                        chat=dict(chat_data),
                        turn=ChatHistoryTurn(human=validated_human_entry, ai=validated_ai_entry)
                    )

                    yield format_sse(json.dumps({"type": "stream_complete", "v": final_response.model_dump_json()}),
                                     event="message")
                except Exception as stream_error:
                    yield format_sse(json.dumps({"error": str(stream_error)}), event="error")

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking chatbot: {str(e)}")
