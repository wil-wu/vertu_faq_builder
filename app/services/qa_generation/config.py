from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


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
                "question_condition": "agent q(?!.*戒指)",
                "answer_condition": "",
                "intent_condition": "产品&功能咨询|产品&品类咨询",
            },
        ],
        description="过滤规则",
    )

qa_generation_service_settings = QAGenerationServiceSettings()
