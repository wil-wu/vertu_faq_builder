from fastapi import Request

from .service import QAGenerationService
from .config import qa_generation_service_settings


def get_qa_generation_service(request: Request) -> QAGenerationService:
    return QAGenerationService(
        request.app.state.openai_client,
        request.app.state.sentence_transformer,
        qa_generation_service_settings.llm_model,
        qa_generation_service_settings.generator_temperature,
        qa_generation_service_settings.filter_temperature,
        qa_generation_service_settings.semantic_threshold,
        qa_generation_service_settings.filter_rules,
    )
