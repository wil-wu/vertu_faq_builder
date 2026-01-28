"""答案增强服务路由"""

from fastapi import APIRouter, Depends

from .models import AnswerEnhancementBody
from .service import AnswerEnhancementService
from .deps import get_answer_enhancement_service

# 创建路由器 - 支持 API 版本化
router = APIRouter(
    prefix="/api/v1/answer",
    tags=["Answer Enhancement"],
)


@router.post("/enhance")
async def answer_enhancement(
    body: AnswerEnhancementBody | list[AnswerEnhancementBody],
    answer_enhancement_service: AnswerEnhancementService = Depends(
        get_answer_enhancement_service
    ),
) -> dict:
    """答案增强"""
    enhanced_answers = []

    if isinstance(body, list):
        for item in body:
            enhanced_answer = await answer_enhancement_service.execute(
                question=item.question, answer=item.answer
            )
            enhanced_answers.append(enhanced_answer)
    else:
        enhanced_answer = await answer_enhancement_service.execute(
            question=body.question, answer=body.answer
        )
        enhanced_answers.append(enhanced_answer)

    return {
        "code": 200,
        "message": "success",
        "data": {
            "total": len(enhanced_answers),
            "enhanced_answers": enhanced_answers,
        },
    }
