import logging

from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI

from .generators import LLMQAGenerator
from .filters import RuleFilter, LLMFilter
from .processors import SemanticProcessor

logger = logging.getLogger(__name__)


class QAGenerationService:
    """问题生成服务"""

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        sentence_transformer: SentenceTransformer,
        llm_model: str,
        generator_temperature: float,
        filter_temperature: float,
        semantic_threshold: float,
        filter_rules: list[dict],
    ):
        """初始化问题生成服务"""
        self.generator_pipeline = [
            LLMQAGenerator(openai_client, llm_model, generator_temperature),
        ]
        self.filter_pipeline = [
            RuleFilter(filter_rules),
            LLMFilter(openai_client, llm_model, filter_temperature),
        ]
        self.post_process_pipeline = [
            # SemanticProcessor(sentence_transformer, semantic_threshold),
        ]

    async def _generate(self, context: str) -> list[dict]:
        """生成QA对"""
        for generator in self.generator_pipeline:
            qa_pairs = await generator.generate(context)
            if qa_pairs:
                return qa_pairs
        return []

    async def _filter(self, qa_pair: dict) -> bool:
        """过滤QA对"""
        for filter in self.filter_pipeline:
            if not await filter.filter(qa_pair):
                return False
        return True

    async def _post_process(self, qas: list[dict]) -> list[dict]:
        """后处理QA对"""
        for processor in self.post_process_pipeline:
            qas = await processor.process(qas)
        return qas

    async def generate_qa(self, contexts: list[str]) -> list[dict]:
        """生成并处理QA对"""
        qas = []

        # 生成候选QA对
        for context in contexts:
            qa_pairs = await self._generate(context)
            qas.extend(qa_pairs)
        logger.info(f"{self.__class__.__name__} generated qas: {len(qas)}")

        # 过滤候选QA对
        qas = [qa_pair for qa_pair in qas if await self._filter(qa_pair)]
        logger.info(f"{self.__class__.__name__} filtered qas: {len(qas)}")

        # 后处理候选QA对
        qas = await self._post_process(qas)
        logger.info(f"{self.__class__.__name__} post processed qas: {len(qas)}")

        return qas
