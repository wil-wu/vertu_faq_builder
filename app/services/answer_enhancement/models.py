from pydantic import BaseModel


class AnswerEnhancementBody(BaseModel):
    question: str
    chat_history: list[dict]
    rag_context: str
