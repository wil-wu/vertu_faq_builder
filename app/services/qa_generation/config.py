from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from .enum import Intent, ProductType


class QAGenerationServiceSettings(BaseSettings):
    """问题生成服务配置"""

    model_config = SettingsConfigDict(
        env_prefix="QA_GENERATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_model: str = Field(
        default="kimi-k2-turbo-preview",
        description="LLM模型",
    )
    generator_temperature: float = Field(default=0.3, description="生成器温度")
    filter_temperature: float = Field(default=0.01, description="过滤器温度")
    semantic_threshold: float = Field(default=0.95, description="语义阈值")
    filter_rules: list[dict] = Field(
        default=[
            {
                "question_condition": f"({'|'.join(ProductType.get_product_types_values())})(?!.*戒指)",
                "answer_condition": "",
                "intent_condition": f"{Intent.PRODUCT_FUNCTION.value}|{Intent.PRODUCT_CATEGORY.value}",
            },
        ],
        description="过滤规则",
    )
    max_context_length: int = Field(default=32 * 1024, description="最大上下文长度")


qa_generation_service_settings = QAGenerationServiceSettings()
