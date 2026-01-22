from fastapi import Request

from .service import AnswerEnhancementService
from .config import enhancement_service_settings


def get_answer_enhancement_service(request: Request) -> AnswerEnhancementService:
    return AnswerEnhancementService(
        request.app.state.openai_client, enhancement_service_settings.llm_model
    )
