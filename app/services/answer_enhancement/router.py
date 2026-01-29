"""答案增强服务路由"""

import orjson
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

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
    return_file: bool = Query(default=False, description="是否返回文件"),
    answer_enhancement_service: AnswerEnhancementService = Depends(
        get_answer_enhancement_service
    ),
) -> Response:
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

    content = orjson.dumps(
        {
            "code": 200,
            "message": "success",
            "data": {
                "total": len(enhanced_answers),
                "enhanced_answers": enhanced_answers,
            },
        }
    )

    if return_file:
        filename = f"enhanced_answers_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    else:
        return Response(content=content, media_type="application/json")
