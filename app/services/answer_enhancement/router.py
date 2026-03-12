"""答案增强服务路由"""

import orjson
from datetime import datetime

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response

from app.core.managers import async_job_manager
from app.core.enum import JobType
from .jobs import enhance_answer
from .models import AnswerEnhancementRequestAdapter
from .service import AnswerEnhancementService
from .deps import get_answer_enhancement_service

# 创建路由器 - 支持 API 版本化
router = APIRouter(
    prefix="/api/v1/answer",
    tags=["Answer Enhancement"],
)


@router.post("/sync/enhance")
async def answer_enhancement(
    request: Request,
    return_file: bool = Query(default=False, description="是否返回文件"),
    answer_enhancement_service: AnswerEnhancementService = Depends(
        get_answer_enhancement_service
    ),
) -> Response:
    """答案增强"""
    body = AnswerEnhancementRequestAdapter.validate_json(await request.body())
    enhanced_answers = []

    if isinstance(body, list):
        for item in body:
            enhanced_answer = await answer_enhancement_service.execute(
                question=item["question"], answer=item["answer"]
            )
            enhanced_answers.append(enhanced_answer)
    else:
        enhanced_answer = await answer_enhancement_service.execute(
            question=body["question"], answer=body["answer"]
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


@router.post("/async/enhance")
async def answer_enhancement_async(
    request: Request,
    answer_enhancement_service: AnswerEnhancementService = Depends(
        get_answer_enhancement_service
    ),
) -> dict:
    """答案增强异步"""
    body = AnswerEnhancementRequestAdapter.validate_json(await request.body())
    job_id = await async_job_manager.create_async_job(
        JobType.ANSWER_ENHANCEMENT, enhance_answer, body, answer_enhancement_service
    )
    return {
        "code": 200,
        "message": "success",
        "data": {"job_id": job_id},
    }
