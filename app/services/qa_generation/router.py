import orjson
from datetime import datetime

from fastapi import APIRouter, Depends, UploadFile, Query
from fastapi.responses import Response

from .service import QAGenerationService
from .deps import get_qa_generation_service
from .models import QAGenerationBody
from .config import qa_generation_service_settings

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
        if not isinstance(contents, list):
            continue
        context = "\n".join(
            [
                f"{idx + 1}. {content.get('sender', '').replace('\n', '')}: {content.get('content', '').replace('\n', '')}"
                for idx, content in enumerate(contents)
            ]
        )
        if len(context) > qa_generation_service_settings.max_context_length:
            context = context[:qa_generation_service_settings.max_context_length]
        contexts.append(context)

    qas = await qa_generation_service.generate_qa(contexts)
    for qa_pair in qas:
        qa_pair["metadata"] = metadata
    return qas


@router.post("/generate_from_body", response_model=None)
async def generate_qa_from_body(
    body: QAGenerationBody,
    return_file: bool = Query(default=False, description="是否返回文件"),
    qa_generation_service: QAGenerationService = Depends(get_qa_generation_service),
) -> Response:
    """从Body生成QA"""

    records = body.data.get("RECORDS", [])
    metadata = body.metadata
    if not metadata:
        metadata = {
            "source": "http request",
            "datetime": datetime.now().isoformat(),
        }

    qas = await _generate_qa(records, metadata, qa_generation_service)
    content = orjson.dumps({
        "code": 200,
        "message": "success",
        "data": {"qas": qas, "total": len(qas)},
    })

    if return_file:
        filename = f"qa_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        return Response(
            content=content, 
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    else:
        return Response(content=content, media_type="application/json")


@router.post("/generate_from_file", response_model=None)
async def generate_qa_from_file(
    file: UploadFile,
    return_file: bool = Query(default=False, description="是否返回文件"),
    qa_generation_service: QAGenerationService = Depends(get_qa_generation_service),
) -> Response:
    """从文件生成QA"""

    records = orjson.loads(await file.read()).get("RECORDS", [])
    metadata = {
        "source": file.filename,
        "datetime": datetime.now().isoformat(),
    }

    qas = await _generate_qa(records, metadata, qa_generation_service)
    content = orjson.dumps({
        "code": 200,
        "message": "success",
        "data": {"qas": qas, "total": len(qas)},
    })

    if return_file:
        filename = f"qa_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        return Response(
            content=content, 
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    else:
        return Response(content=content, media_type="application/json")
