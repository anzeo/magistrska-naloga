from typing import Optional

from pydantic import BaseModel


class ChatUpdate(BaseModel):
    name: str


class InvokeChatbotRequestBody(BaseModel):
    chat_id: Optional[str] = None
    user_input: str


class InvokeChatbotResponse(BaseModel):
    chat_id: str
    user_input: str
    answer: str
    relevant_part_texts: Optional[list] = []