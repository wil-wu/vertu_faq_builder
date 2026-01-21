from openai import AsyncOpenAI

from .checkers import LLMChecker
from .enhancers import LLMEnhancer
from .rewriters import LLMQueryRewriter
from .enum import EnhancementStrategy


class AnswerEnhancementService:
    """答案增强服务"""

    def __init__(self, openai_client: AsyncOpenAI, llm_model: str):
        """初始化答案增强服务"""
        self.check_pipeline = [LLMChecker(openai_client, llm_model)]
        self.enhance_pipeline = [LLMEnhancer(openai_client, llm_model)]

    async def _check(self, question: str, answer: str) -> EnhancementStrategy:
        """策略判断"""
        for checker in self.check_pipeline:
            strategy = await checker.check(question, answer)
            if strategy:
                return strategy
        return EnhancementStrategy.DIRECT

    async def _enhance(
        self, question: str, answer: str, strategy: EnhancementStrategy
    ) -> str:
        """根据策略增强答案"""
        for enhancer in self.enhance_pipeline:
            enhanced_answer = await enhancer.enhance(question, answer, strategy)
            if enhanced_answer:
                return enhanced_answer
        return answer

    async def execute(self, question: str, answer: str) -> str:
        """策略判断并增强答案"""
        strategy = await self._check(question, answer)
        enhanced_answer = await self._enhance(question, answer, strategy)
        return enhanced_answer


class QueryRewritingService:
    """问题重写服务"""

    def __init__(self, openai_client: AsyncOpenAI, llm_model: str):
        """初始化问题重写服务"""
        self.rewrite_pipeline = [LLMQueryRewriter(openai_client, llm_model)]

    def _flatten_chat_history(self, chat_history: list[dict]) -> str:
        """扁平化历史对话"""
        return "\n".join(
            [
                f"{idx + 1}. {item['role']}: {item['content']}"
                for idx, item in enumerate(chat_history)
            ]
        )

    async def execute(
        self, question: str, chat_history: list[dict], rag_context: str
    ) -> str:
        """重写问题"""
        flattened_chat_history = self._flatten_chat_history(chat_history)
        for rewriter in self.rewrite_pipeline:
            rewritten_query = await rewriter.rewrite(
                question, flattened_chat_history, rag_context
            )
            if rewritten_query:
                return rewritten_query
        return question
