from pydantic import BaseModel


class AnswerEnhancementBody(BaseModel):
    question: str
    answer: str
