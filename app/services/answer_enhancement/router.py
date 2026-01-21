"""答案增强服务路由"""

from fastapi import APIRouter, Depends, Request

from .models import AnswerEnhancementBody
from .service import AnswerEnhancementService, QueryRewritingService
from .deps import get_answer_enhancement_service, get_query_rewriting_service
from .utils import get_faq_answer

# 创建路由器 - 支持 API 版本化
router = APIRouter(
    prefix="/api/v1/answer",
    tags=["Answer Enhancement"],
)


@router.post("/enhance")
async def answer_enhancement(
    request: Request,
    body: AnswerEnhancementBody,
    answer_enhancement_service: AnswerEnhancementService = Depends(
        get_answer_enhancement_service
    ),
    query_rewriting_service: QueryRewritingService = Depends(
        get_query_rewriting_service
    ),
) -> dict:
    """答案增强"""
    rewritten_question = await query_rewriting_service.execute(
        body.question, body.chat_history, body.rag_context
    )

    faq_answer = await get_faq_answer(request.app.state.httpx_client, rewritten_question)
    enhanced_answer = await answer_enhancement_service.execute(
        rewritten_question, faq_answer
    )

    return {
        "code": 200,
        "message": "success",
        "data": {"enhanced_answer": enhanced_answer},
    }
