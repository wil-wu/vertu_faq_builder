import logging

from app.core.managers import async_job_manager
from app.core.enum import JobStatus
from .service import QAGenerationService
from .utils import build_contexts
from .models import ChatSession

logger = logging.getLogger(__name__)


async def generate_qa(
    job_id: str,
    chat_sessions: list[ChatSession],
    metadata: dict,
    service: QAGenerationService,
) -> None:
    """QA 生成任务"""
    try:
        contexts = build_contexts(chat_sessions)
        filtered_qas = []
        generated_count = 0
        progress = 0

        for idx, context in enumerate(contexts):
            qa_pairs = await service._generate(context)
            generated_count += len(qa_pairs)
            for qa_pair in qa_pairs:
                if await service._filter(qa_pair):
                    filtered_qas.append(qa_pair)

            _progress = int((idx + 1) / len(contexts) * 100)
            if _progress > progress:
                progress = _progress
                await async_job_manager.update_async_job(job_id, progress=progress)

        post_processed_qas = await service._post_process(filtered_qas)

        for qa_pair in post_processed_qas:
            qa_pair["metadata"] = metadata

        qas_result = {
            "generated_count": generated_count,
            "filtered_count": len(filtered_qas),
            "post_processed_count": len(post_processed_qas),
            "total": len(post_processed_qas),
            "qas": post_processed_qas,
        }

        await async_job_manager.update_async_job(
            job_id, status=JobStatus.COMPLETED, result=qas_result
        )
    except Exception as e:
        logger.exception("QA generation job %s failed", job_id, exc_info=True)
        await async_job_manager.update_async_job(
            job_id, status=JobStatus.FAILED, error=str(e)
        )
