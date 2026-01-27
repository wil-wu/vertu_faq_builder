from typing import Optional

from pydantic import BaseModel, Field


class QAGenerationBody(BaseModel):
    data: dict
    metadata: Optional[dict] = Field(default=None)
