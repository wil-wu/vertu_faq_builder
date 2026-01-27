import json
from datetime import datetime

from fastapi import APIRouter, Depends, UploadFile, Query
from fastapi.responses import StreamingResponse

from .service import QAGenerationService
from .deps import get_qa_generation_service
from .models import QAGenerationBody

router = APIRouter(
    prefix="/api/v1/qa",
    tags=["QA Generation"],
)


async def _generate_qa(
    records: list[dict], metadata: dict, qa_generation_service: QAGenerationService
) -> list[dict]:
    """生成QA"""
    contexts = []
    for record in records:
        contents = record.get("消息内容", [])
        context = "\n".join(
            [
                f"{content.get('sender', '')}: {content.get('content', '')}"
                for content in contents
            ]
        )
        contexts.append(context)

    qas = await qa_generation_service.generate_qa(contexts)
    for qa_pair in qas:
        qa_pair["metadata"] = metadata
    return qas


@router.post("/generate_from_body", response_model=None)
async def generate_qa_from_body(
    body: QAGenerationBody,
    is_stream: bool = Query(default=False, description="是否流式返回"),
    qa_generation_service: QAGenerationService = Depends(get_qa_generation_service),
) -> dict | StreamingResponse:
    """从Body生成QA"""

    records = body.data.get("RECORDS", [])
    metadata = body.metadata
    if not metadata:
        metadata = {
            "source": "http request",
            "datetime": datetime.now().isoformat(),
        }

    qas = await _generate_qa(records, metadata, qa_generation_service)

    if is_stream:
        return StreamingResponse(
            content=json.dumps(qas, ensure_ascii=False),
            media_type="application/json",
        )
    else:
        return {
            "code": 200,
            "message": "success",
            "data": {"qas": qas, "total": len(qas)},
        }


@router.post("/generate_from_file", response_model=None)
async def generate_qa_from_file(
    file: UploadFile,
    is_stream: bool = Query(default=False, description="是否流式返回"),
    qa_generation_service: QAGenerationService = Depends(get_qa_generation_service),
) -> dict | StreamingResponse:
    """从文件生成QA"""

    records = json.loads(await file.read()).get("RECORDS", [])
    metadata = {
        "source": file.filename,
        "datetime": datetime.now().isoformat(),
    }

    qas = await _generate_qa(records, metadata, qa_generation_service)

    if is_stream:
        return StreamingResponse(
            content=json.dumps(qas, ensure_ascii=False),
            media_type="application/json",
        )
    else:
        return {
            "code": 200,
            "message": "success",
            "data": {"qas": qas, "total": len(qas)},
        }
