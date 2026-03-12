import logging

from app.core.managers import async_job_manager
from app.core.enum import JobStatus
from .service import AnswerEnhancementService
from .models import AnswerEnhancementRequest

logger = logging.getLogger(__name__)


async def enhance_answer(
    job_id: str,
    body: AnswerEnhancementRequest | list[AnswerEnhancementRequest],
    service: AnswerEnhancementService,
) -> None:
    """答案增强任务"""
    try:
        enhanced_answers = []

        if isinstance(body, list):
            total = len(body)
            progress = 0
            for idx, item in enumerate(body):
                enhanced_answer = await service.execute(
                    question=item["question"], answer=item["answer"]
                )
                enhanced_answers.append(enhanced_answer)
                _progress = int((idx + 1) / total * 100)
                if _progress > progress:
                    progress = _progress
                    await async_job_manager.update_async_job(
                        job_id,
                        progress=progress,
                    )
        else:
            enhanced_answer = await service.execute(
                question=body["question"], answer=body["answer"]
            )
            enhanced_answers.append(enhanced_answer)
            progress = 100
            await async_job_manager.update_async_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress=progress,
                result={
                    "total": len(enhanced_answers),
                    "enhanced_answers": enhanced_answers,
                },
            )

    except Exception as e:
        logger.exception("Answer enhancement job %s failed", job_id, exc_info=True)
        await async_job_manager.update_async_job(
            job_id, status=JobStatus.FAILED, error=str(e)
        )
