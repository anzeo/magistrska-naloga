from typing import Optional, Literal, Annotated, Union, Dict, Any

from pydantic import BaseModel, Field, model_validator


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
    parent_id: str
    content: str
    relevant_part_texts: Optional[list] = Field(default_factory=list)

    @model_validator(mode="before")
    def extract_relevant_parts(self):
        # Move relevant_part_texts from response_metadata to top-level
        response_metadata = self.get("response_metadata", {})
        relevant = response_metadata.get("relevant_part_texts", [])
        self["relevant_part_texts"] = relevant
        return self

    @model_validator(mode="before")
    def extract_parent_id(self):
        # Move parent_id from additional_kwargs to top-level
        additional_kwargs = self.get("additional_kwargs", {})
        parent_id = additional_kwargs.get("parent_id", None)
        self["parent_id"] = parent_id
        return self


ChatHistoryEntry = Annotated[
    Union[ChatHistoryHumanEntry, ChatHistoryAIEntry],
    Field(discriminator="type")
]


# ========== Request/Response Models ==========

class ChatUpdate(BaseModel):
    name: str


class ChatHistoryTurn(BaseModel):
    human: ChatHistoryEntry
    ai: ChatHistoryEntry


class InvokeChatbotRequestBody(BaseModel):
    chat_id: Optional[str] = None
    user_input: str


class InvokeChatbotStreamingResponse(BaseModel):
    chat: Dict[str, Any]
    turn: ChatHistoryTurn
