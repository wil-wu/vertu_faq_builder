from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnswerEnhancementSettings(BaseSettings):
    """答案增强服务配置"""

    model_config = SettingsConfigDict(
        env_prefix="ANSWER_ENHANCEMENT_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # 模型配置
    llm_model: str = Field(default="kimi-k2-turbo-preview", description="LLM模型")


enhancement_service_settings = AnswerEnhancementSettings()
