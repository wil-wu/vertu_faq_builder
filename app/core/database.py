from datetime import datetime

import orjson
from sqlalchemy import DateTime, JSON, TypeDecorator, inspect
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings
from app.core.enum import JobStatus, JobType

async_engine = create_async_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)

async_session = async_sessionmaker(
    async_engine,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


class OrJSON(TypeDecorator):
    """ORJSON类型增强"""

    impl = JSON
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return orjson.dumps(value).decode("utf-8") if value else None

    def process_result_value(self, value, dialect):
        return orjson.loads(value) if value else None


class LocalDatetime(TypeDecorator):
    """本地时间类型增强"""

    impl = DateTime
    cache_ok = True

    def process_result_value(self, value, dialect):
        return value.strftime("%Y-%m-%d %H:%M:%S") if value else None


class LoadOnlyDictMixin:
    """只返回已加载的属性"""

    def to_dict(self):
        inst = inspect(self)

        return {
            column.key: getattr(self, column.key)
            for column in inst.mapper.column_attrs
            if column.key not in inst.unloaded
        }


class Job(Base, LoadOnlyDictMixin):
    """异步任务模型"""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(unique=True, index=True, nullable=False)
    job_type: Mapped[JobType] = mapped_column(nullable=False, default=JobType.UNKNOWN)
    status: Mapped[JobStatus] = mapped_column(nullable=False, default=JobStatus.PENDING)
    progress: Mapped[int] = mapped_column(nullable=False, default=0)
    result: Mapped[dict | None] = mapped_column(OrJSON, nullable=True)
    error: Mapped[str | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(LocalDatetime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        LocalDatetime, default=datetime.now, onupdate=datetime.now
    )
