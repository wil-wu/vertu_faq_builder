from enum import Enum


class JobStatus(Enum):
    """任务状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class JobType(Enum):
    """任务类型"""

    UNKNOWN = "unknown"
    QA_GENERATION = "qa_generation"
    ANSWER_ENHANCEMENT = "answer_enhancement"
