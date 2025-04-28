from pydantic import BaseModel


class ChatUpdate(BaseModel):
    name: str
