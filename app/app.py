"""FastAPI 应用入口"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import orjson
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from openai import AsyncOpenAI
from httpx import AsyncClient
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.scanner import RouterScanner
from app.core.middlewares import RequestLoggingMiddleware
from app.core.database import Base, async_engine
from app.core.managers import async_job_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    # 启动时执行
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # 初始化数据库表
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    app.state.openai_client = AsyncOpenAI(
        api_key=settings.openai_api_key, base_url=settings.openai_base_url
    )
    app.state.httpx_client = AsyncClient()
    app.state.sentence_transformer = SentenceTransformer(
        settings.sentence_transformer_model
    )

    logger.info("Application startup completed")

    yield

    # 关闭时执行
    logger.info("Shutting down application")

    await app.state.openai_client.close()
    await app.state.httpx_client.aclose()
    await async_engine.dispose()

    logger.info("Application shutdown completed")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )

    # 配置请求日志中间件
    app.add_middleware(
        RequestLoggingMiddleware,
        log_request_body=settings.debug,  # 仅在调试模式下记录请求体
        log_request_body_length=settings.log_request_body_length,  # 请求体日志长度
        exclude_paths=["/health", "/metrics"],  # 排除健康检查和指标端点
    )

    # 配置 Prometheus 监控
    if settings.enable_metrics:
        instrumentator = Instrumentator(
            should_group_status_codes=False,
            should_ignore_untemplated=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/metrics", "/health"],
            inprogress_name="fastapi_inprogress",
            inprogress_labels=True,
        )
        instrumentator.instrument(app).expose(app, endpoint=settings.metrics_path)
        logger.info(f"Metrics enabled at {settings.metrics_path}")

    # 注册所有服务路由
    scanner = RouterScanner(app)
    scanner.scan_and_register()

    # 注册全局路由
    @app.get("/", tags=["Root"])
    async def root():
        """根路径"""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "running",
        }

    @app.get("/health", tags=["Health"])
    async def health_check():
        """健康检查"""
        return {
            "status": "healthy",
            "app_name": settings.app_name,
            "version": settings.app_version,
        }

    @app.get("/routes", tags=["Debug"])
    async def list_routes():
        """列出所有路由(仅调试模式)"""
        if not settings.debug:
            return JSONResponse(
                status_code=403,
                content={"detail": "This endpoint is only available in debug mode"},
            )

        routes = scanner.get_registered_routes()
        return {"total": len(routes), "routes": routes}

    @app.get("/jobs/{job_id}", tags=["Jobs"])
    async def get_async_job(job_id: str) -> Response:
        """根据任务 ID 获取单个异步任务详情。

        ```
        Args:
            job_id: 任务唯一标识（UUID）。

        Returns:
            Response: 成功时返回任务详情，不存在时返回 404。

        Response body (JSON schema, 成功 200):
            {
                "code": 200,
                "message": "success",
                "data": {
                    "job_id": "string",
                    "job_type": "qa_generation | answer_enhancement | unknown",
                    "status": "pending | running | completed | cancelled | failed",
                    "progress": 0,
                    "result": {} | null,
                    "error": "string | null",
                    "created_at": "string (ISO datetime)",
                    "updated_at": "string (ISO datetime)"
                }
            }

        Response body (JSON schema, 不存在 404):
            {"code": 404, "message": "Job not found", "data": null}
        ```
        """
        job = await async_job_manager.get_async_job(job_id)
        if job is None:
            return Response(
                content=orjson.dumps(
                    {"code": 404, "message": "Job not found", "data": None}
                ),
                media_type="application/json",
                status_code=404,
            )

        return Response(
            content=orjson.dumps({"code": 200, "message": "success", "data": job}),
            media_type="application/json",
        )

    @app.get("/jobs", tags=["Jobs"])
    async def get_async_jobs(
        page: int = 1,
        size: int = 10,
        with_result: bool = False,
    ) -> Response:
        """分页获取异步任务列表。

        ```
        Args:
            page: 页码，从 1 开始。
            size: 每页条数。
            with_result: 为 True 时返回每条任务的 result 字段，否则不查该字段以节省开销。

        Returns:
            Response: 分页结果。

        Response body (JSON schema):
            {
                "code": 200,
                "message": "success",
                "data": {
                    "items": [
                        {
                            "job_id": "string",
                            "job_type": "string",
                            "status": "string",
                            "progress": 0,
                            "result": {} | null,
                            "error": "string | null",
                            "created_at": "string",
                            "updated_at": "string"
                        }
                    ],
                    "total": 0,
                    "page": 1,
                    "size": 10
                }
            }
        ```
        """
        jobs = await async_job_manager.get_async_jobs(
            page=page, size=size, with_result=with_result
        )
        return Response(
            content=orjson.dumps({"code": 200, "message": "success", "data": jobs}),
            media_type="application/json",
        )

    @app.get("/jobs/{job_id}/cancel", tags=["Jobs"])
    async def cancel_async_job(job_id: str) -> Response:
        """取消指定 ID 的异步任务（仅对运行中任务生效）。

        ```
        Args:
            job_id: 任务唯一标识（UUID）。

        Returns:
            Response: 取消成功返回 200，任务不存在或无法取消返回 400。

        Response body (JSON schema, 成功 200):
            {"code": 200, "message": "success", "data": null}

        Response body (JSON schema, 失败 400):
            {"code": 400, "message": "Job not found", "data": null}
        ```
        """
        cancelled = await async_job_manager.cancel_async_job(job_id)
        if not cancelled:
            return JSONResponse(
                status_code=400,
                content={"code": 400, "message": "Job not found", "data": None},
            )

        return JSONResponse(
            status_code=200, content={"code": 200, "message": "success", "data": None}
        )

    logger.info("FastAPI application initialized")

    return app
