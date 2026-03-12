import orjson
from datetime import datetime

from fastapi import APIRouter, Depends, UploadFile, Query, Request
from fastapi.responses import Response

from app.core.managers import async_job_manager
from app.core.enum import JobType
from .jobs import generate_qa
from .service import QAGenerationService
from .deps import get_qa_generation_service
from .models import ChatSession, QAGenerationRequestAdapter
from .utils import build_contexts

router = APIRouter(
    prefix="/api/v1/qa",
    tags=["QA Generation"],
)


async def _generate_qa(
    chat_sessions: list[ChatSession], metadata: dict, qa_generation_service: QAGenerationService
) -> list[dict]:
    """生成QA"""
    contexts = build_contexts(chat_sessions)

    qas_result = await qa_generation_service.generate_qa(contexts)
    for qa_pair in qas_result["qas"]:
        qa_pair["metadata"] = metadata
    return qas_result


@router.post("/sync/generate_from_body")
async def generate_qa_from_body(
    request: Request,
    return_file: bool = Query(default=False, description="是否返回文件"),
    qa_generation_service: QAGenerationService = Depends(get_qa_generation_service),
) -> Response:
    """从请求体中的会话数据同步生成 QA。

    Args:
        request: FastAPI 请求对象。
        return_file: 为 True 时以文件形式返回结果，否则返回 JSON 响应。

    Request body (JSON schema):
        {
            "data": [
                {
                    "messages": [
                        {
                            "role": "string",
                            "content": "string",
                            "datetime": "string (optional)"
                        }
                    ]
                }
            ],
            "metadata": {}  // optional, arbitrary key-value
        }

    Returns:
        Response: JSON 或文件下载响应。

    Response body (JSON schema):
        {
            "code": 200,
            "message": "success",
            "data": {
                "generated_count": 0,
                "filtered_count": 0,
                "post_processed_count": 0,
                "total": 0,
                "qas": [{"question": "...", "answer": "...", "metadata": {}}]
            }
        }
    """
    body = QAGenerationRequestAdapter.validate_json(await request.body())
    chat_sessions = body["data"]
    metadata = body.get("metadata")
    if not metadata:
        metadata = {
            "source": "http request",
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    qas_result = await _generate_qa(chat_sessions, metadata, qa_generation_service)
    content = orjson.dumps(
        {
            "code": 200,
            "message": "success",
            "data": qas_result,
        }
    )

    if return_file:
        filename = f"qa_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    else:
        return Response(content=content, media_type="application/json")


@router.post("/sync/generate_from_file")
async def generate_qa_from_file(
    file: UploadFile,
    return_file: bool = Query(default=False, description="是否返回文件"),
    qa_generation_service: QAGenerationService = Depends(get_qa_generation_service),
) -> Response:
    """从上传的 JSON 文件中的会话数据同步生成 QA。

    Args:
        file: 上传的 JSON 文件。
        return_file: 为 True 时以文件形式返回结果，否则返回 JSON 响应。

    Request body (file content, JSON schema):
        {
            "data": [
                {
                    "messages": [
                        {
                            "role": "string",
                            "content": "string",
                            "datetime": "string (optional)"
                        }
                    ]
                }
            ],
            "metadata": {}  // optional, arbitrary key-value
        }

    Returns:
        Response: JSON 或文件下载响应。

    Response body (JSON schema):
        {
            "code": 200,
            "message": "success",
            "data": {
                "generated_count": 0,
                "filtered_count": 0,
                "post_processed_count": 0,
                "total": 0,
                "qas": [{"question": "...", "answer": "...", "metadata": {}}]
            }
        }
    """
    body = QAGenerationRequestAdapter.validate_json(await file.read())
    chat_sessions = body["data"]
    metadata = body.get("metadata")
    if not metadata:
        metadata = {
            "source": file.filename,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    qas_result = await _generate_qa(chat_sessions, metadata, qa_generation_service)
    content = orjson.dumps(
        {
            "code": 200,
            "message": "success",
            "data": qas_result,
        }
    )

    if return_file:
        filename = f"qa_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    else:
        return Response(content=content, media_type="application/json")


@router.post("/async/generate_from_body")
async def generate_qa_from_body_async(
    request: Request,
    qa_generation_service: QAGenerationService = Depends(get_qa_generation_service),
) -> dict:
    """从请求体中的会话数据异步生成 QA，返回任务 ID。

    Args:
        request: FastAPI 请求对象。

    Request body (JSON schema):
        {
            "data": [
                {
                    "messages": [
                        {
                            "role": "string",
                            "content": "string",
                            "datetime": "string (optional)"
                        }
                    ]
                }
            ],
            "metadata": {}  // optional, arbitrary key-value
        }

    Returns:
        dict: 响应体。

    Response body (JSON schema):
        {"code": 200, "message": "success", "data": {"job_id": "string"}}
    """
    body = QAGenerationRequestAdapter.validate_json(await request.body())
    chat_sessions = body["data"]
    metadata = body.get("metadata")
    if not metadata:
        metadata = {
            "source": "http request",
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    job_id = await async_job_manager.create_async_job(
        JobType.QA_GENERATION, generate_qa, chat_sessions, metadata, qa_generation_service
    )

    return {
        "code": 200,
        "message": "success",
        "data": {"job_id": job_id},
    }


@router.post("/async/generate_from_file")
async def generate_qa_from_file_async(
    file: UploadFile,
    qa_generation_service: QAGenerationService = Depends(get_qa_generation_service),
) -> dict:
    """从上传的 JSON 文件中的会话数据异步生成 QA，返回任务 ID。

    Args:
        file: 上传的 JSON 文件。

    Request body (file content, JSON schema):
        {
            "data": [
                {
                    "messages": [
                        {
                            "role": "string",
                            "content": "string",
                            "datetime": "string (optional)"
                        }
                    ]
                }
            ],
            "metadata": {}  // optional, arbitrary key-value
        }

    Returns:
        dict: 响应体。

    Response body (JSON schema):
        {"code": 200, "message": "success", "data": {"job_id": "string"}}
    """
    body = QAGenerationRequestAdapter.validate_json(await file.read())
    chat_sessions = body["data"]
    metadata = body.get("metadata")
    if not metadata:
        metadata = {
            "source": file.filename,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    job_id = await async_job_manager.create_async_job(
        JobType.QA_GENERATION, generate_qa, chat_sessions, metadata, qa_generation_service
    )

    return {
        "code": 200,
        "message": "success",
        "data": {"job_id": job_id},
    }
