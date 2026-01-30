import logging

from openai import AsyncOpenAI

from .checkers import LLMChecker
from .enhancers import LLMEnhancer
from .extractors import LLMExtractor
from .enum import EnhancementStrategy

logger = logging.getLogger(__name__)


class AnswerEnhancementService:
    """答案增强服务"""

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        llm_model: str,
        checker_temperature: float,
        enhancer_temperature: float,
        extractor_temperature: float,
    ):
        """初始化答案增强服务"""
        self.check_pipeline = [
            LLMChecker(openai_client, llm_model, checker_temperature)
        ]
        self.enhance_pipeline = [
            LLMEnhancer(openai_client, llm_model, enhancer_temperature)
        ]
        self.extract_pipeline = [
            LLMExtractor(openai_client, llm_model, extractor_temperature)
        ]

    async def _check(self, question: str, answer: str) -> EnhancementStrategy:
        """策略判断"""
        for checker in self.check_pipeline:
            strategy = await checker.check(question, answer)
            try:
                strategy = EnhancementStrategy.get_strategy(strategy)
                return strategy
            except ValueError:
                logger.error(
                    f"{checker.__class__.__name__} strategy is not a valid strategy: {strategy}"
                )

        return EnhancementStrategy.DIRECT

    async def _enhance(self, question: str, answer: str, strategy: str) -> str:
        """根据策略增强答案"""
        for enhancer in self.enhance_pipeline:
            enhanced_answer = await enhancer.enhance(question, answer, strategy)
            if enhanced_answer:
                return enhanced_answer
        return answer

    async def _extract(self, question: str, answer: str) -> str:
        """提取图片/视频描述文本"""
        for extractor in self.extract_pipeline:
            extracted_answer = await extractor.extract(question, answer)
            if extracted_answer:
                return extracted_answer
        return ""

    async def execute(self, question: str, answer: str) -> str:
        """策略判断并增强答案"""
        strategy = await self._check(question, answer)
        enhanced_answer = await self._enhance(question, answer, strategy.value)
        if strategy == EnhancementStrategy.GUIDANCE:
            guidance_answer = await self._extract(question, answer)
            return f"{enhanced_answer}[{guidance_answer}]"

        return enhanced_answer
