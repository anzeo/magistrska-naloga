from typing import Optional, Literal, Annotated, Union

from pydantic import BaseModel, Field, model_validator


# ========== Request/Response Models ==========

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


# ========== Chat History Models ==========


class ChatHistoryHumanEntry(BaseModel):
    type: Literal["human"]
    chat_id: str
    id: str
    content: str


class ChatHistoryAIEntry(BaseModel):
    type: Literal["ai"]
    chat_id: str
    id: str
    content: str
    relevant_part_texts: Optional[list] = Field(default_factory=list)

    @model_validator(mode="before")
    def extract_relevant_parts(self):
        # Move relevant_part_texts from response_metadata to top-level
        response_metadata = self.get("response_metadata", {})
        relevant = response_metadata.get("relevant_part_texts", [])
        self["relevant_part_texts"] = relevant
        return self


ChatHistoryEntry = Annotated[
    Union[ChatHistoryHumanEntry, ChatHistoryAIEntry],
    Field(discriminator="type")
]