from typing import TypedDict

from pydantic import TypeAdapter


class AnswerEnhancementRequest(TypedDict):
    question: str
    answer: str


AnswerEnhancementRequestAdapter = TypeAdapter(
    AnswerEnhancementRequest | list[AnswerEnhancementRequest]
)
