import uuid
import asyncio
import logging
from collections.abc import Callable
from typing import Any, Coroutine, Self

from sqlalchemy import func, select, update
from sqlalchemy.orm import load_only

from app.core.database import async_session, Job
from app.core.enum import JobStatus, JobType

logger = logging.getLogger(__name__)


class AsyncJobManager:
    """异步任务管理器"""

    _instance: Self | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # 运行中的异步任务：job_id -> asyncio.Task，用于取消等操作
        self._async_tasks: dict[str, asyncio.Task] = {}

    async def create_async_job(
        self,
        job_type: JobType,
        coro: Callable[..., Coroutine[Any, Any, None]],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """创建异步任务记录"""
        job_id = str(uuid.uuid4())

        async with async_session() as session:
            job = Job(
                job_id=job_id,
                job_type=job_type,
            )
            session.add(job)
            await session.commit()

            try:
                task = asyncio.create_task(coro(job_id, *args, **kwargs))
                self._async_tasks[job_id] = task
                await self.update_async_job(job_id, status=JobStatus.RUNNING)
            except Exception as e:
                logger.exception(
                    "Failed to create async task for job %s", job_id, exc_info=True
                )
                await self.update_async_job(
                    job_id, status=JobStatus.FAILED, error=str(e)
                )
        return job_id

    async def get_async_job(self, job_id: str) -> dict | None:
        """获取异步任务详情"""
        async with async_session() as session:
            result = await session.execute(select(Job).where(Job.job_id == job_id))
            job = result.scalar_one_or_none()
            if job is None:
                return None

            return job.to_dict()

    async def get_async_jobs(
        self, page: int, size: int, with_result: bool, **kwargs: Any
    ) -> dict:
        """获取异步任务详情（分页）
        with_result=True 时查询并返回 result 字段，反之不查
        kwargs 为其他过滤条件，如 job_type、status 等
        """
        # 构建过滤条件
        filters = []
        for key, value in kwargs.items():
            if hasattr(Job, key):
                filters.append(getattr(Job, key) == value)

        # 构建查询
        if with_result:
            stmt = select(Job)
        else:
            stmt = select(Job).options(
                load_only(
                    Job.job_id,
                    Job.job_type,
                    Job.status,
                    Job.progress,
                    Job.error,
                    Job.created_at,
                    Job.updated_at,
                )
            )

        for f in filters:
            stmt = stmt.where(f)
        stmt = stmt.order_by(Job.created_at.desc())

        async with async_session() as session:
            count_stmt = select(func.count(Job.id))
            for f in filters:
                count_stmt = count_stmt.where(f)
            count_result = await session.execute(count_stmt)
            total = count_result.scalar() or 0

            offset = (page - 1) * size
            result = await session.execute(stmt.offset(offset).limit(size))
            jobs = result.scalars().all()

        items = [job.to_dict() for job in jobs]

        return {"items": items, "total": total, "page": page, "size": size}

    async def update_async_job(self, job_id: str, **kwargs: Any) -> None:
        """更新异步任务详情"""
        async with async_session() as session:
            stmt = update(Job).where(Job.job_id == job_id).values(**kwargs)
            await session.execute(stmt)
            await session.commit()

        status = kwargs.get("status")
        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            self._async_tasks.pop(job_id, None)

    async def cancel_async_job(self, job_id: str) -> bool:
        """取消异步任务并更新数据库状态"""
        task = self._async_tasks.pop(job_id, None)
        if task is None:
            return False

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info("Async task %s cancelled", job_id)
        finally:
            await self.update_async_job(job_id, status=JobStatus.CANCELLED)
        return True


# 默认单例，供模块级调用
async_job_manager = AsyncJobManager()
