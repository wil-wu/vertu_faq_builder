from typing import Annotated, NotRequired, Optional, TypedDict

from pydantic import AfterValidator, TypeAdapter


def _normalize_content(x: str) -> str:
    return x.replace("\n", "").replace("\r", "")


class Message(TypedDict):
    role: str
    content: Annotated[str, AfterValidator(_normalize_content)]
    datetime: NotRequired[Optional[str]]


class ChatSession(TypedDict):
    messages: list[Message]


class QAGenerationRequest(TypedDict):
    data: list[ChatSession]
    metadata: NotRequired[Optional[dict]]


MessageAdapter = TypeAdapter(Message)
ChatSessionAdapter = TypeAdapter(ChatSession)
QAGenerationRequestAdapter = TypeAdapter(QAGenerationRequest)