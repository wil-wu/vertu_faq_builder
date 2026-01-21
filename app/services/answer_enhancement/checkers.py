import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncOpenAI

from .enum import EnhancementStrategy

logger = logging.getLogger(__name__)


class Checker(ABC):
    """检查器抽象基类"""

    @abstractmethod
    def check(self, question: str, answer: str) -> EnhancementStrategy:
        """策略判断"""
        raise NotImplementedError


class RuleChecker(Checker):
    """规则检查器"""

    def __init__(self, rules: list[str]):
        """初始化规则检查器"""
        self.rules = rules

    def check(self, question: str, answer: str) -> EnhancementStrategy:
        """策略判断"""
        return False


class MLChecker(Checker):
    """机器学习模型检查器"""

    def __init__(self, model: Any):
        """初始化机器学习模型检查器"""
        self.model = model

    def check(self, question: str, answer: str) -> EnhancementStrategy:
        """策略判断"""
        return False


class LLMChecker(Checker):
    """LLM检查器"""

    system_prompt: str = """
    <role>
    你是一位专业的客服助理,负责分析用户咨询和原始答案,为客服人员选择最佳的回复策略。
    你的目标是帮助客服提供准确、清晰、友好的服务体验。
    </role>
    <task>
    分析用户问题和原始答案,选择最合适的答案增强策略。
    </task>

    <strategies>
    1. DIRECT: 问题明确且原始答案完整有意义,可直接使用
    2. CLARIFICATION: 问题存在歧义或原始答案不足以回答问题,需要反问澄清
    3. GUIDANCE: 原始答案可用但存在更优表达形式(图片/视频),需添加引导语
    </strategies>

    <analysis_steps>
    1. 检查原始答案是否为空或无意义
    - 如为空/无意义 → CLARIFICATION
    2. 检查用户问题是否明确
    - 如问题指代不明、缺少主语 → CLARIFICATION
    3. 检查原始答案是否完整回答问题
    - 如答非所问或信息不足 → CLARIFICATION
    4. 检查原始答案中是否包含图片/视频链接
    - 如包含且缺少引导语 → GUIDANCE
    5. 检查问题类型是否适合图片/视频展示(如:拍照、外观、视频、演示等)
    - 如适合且原始答案纯文本 → GUIDANCE
    6. 其他情况 → DIRECT
    </analysis_steps>

    <output_format>
    输出JSON格式,包含策略名称和决策理由:
    {
    "strategy": "策略名称",
    "reason": "简要决策原因"
    }

    注意:
    - strategy值只能是: DIRECT, CLARIFICATION, GUIDANCE
    - reason需简洁说明选择该策略的关键原因(不超过30字)
    - 不要输出任何JSON之外的内容
    </output_format>

    <examples>
    <example>
    用户问题: 苹果耳机跟你们音质最好的耳机对比哪个好?
    原始答案: 作为音质巅峰,VERTU耳机融入伦敦交响乐团专属调校...
    输出: {
    "strategy": "DIRECT",
    "reason": "问题明确,原始答案完整回答了对比问题"
    }
    </example>

    <example>
    用户问题: 我觉得这个手机还不错
    原始答案: 
    输出: {
    "strategy": "CLARIFICATION",
    "reason": "问题指代不明且原始答案为空"
    }
    </example>

    <example>
    用户问题: 拍照怎么样?
    原始答案: [QuantumFlip实拍图]
    输出: {
    "strategy": "GUIDANCE",
    "reason": "原始答案仅含图片链接,缺少引导语"
    }
    </example>

    <example>
    用户问题: 拍照怎么样?
    原始答案: QuantumFlip后置5000万AI双摄,支持双重防抖。AI暗房师功能配合前置3200万镜头,自拍更立体。
    输出: {
    "strategy": "GUIDANCE",
    "reason": "拍照问题适合图片展示,需补充实拍图"
    }
    </example>

    <example>
    用户问题: 电池续航怎么样?
    原始答案: 这款手机很不错
    输出: {
    "strategy": "CLARIFICATION",
    "reason": "原始答案答非所问,未回答续航问题"
    }
    </example>

    <example>
    用户问题: 有什么颜色?
    原始答案: 
    输出: {
    "strategy": "CLARIFICATION",
    "reason": "原始答案为空,需询问具体产品"
    }
    </example>
    </examples>
    """

    user_prompt: str = """
    <input>
    - 用户问题: {question}
    - 原始答案: {answer}
    </input>
    """

    def __init__(self, openai_client: AsyncOpenAI, llm_model: str):
        """初始化LLM检查器"""
        self.client = openai_client
        self.llm_model = llm_model

    async def check(self, question: str, answer: str) -> EnhancementStrategy:
        """策略判断"""
        response = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        question=question,
                        answer=answer,
                    ),
                },
            ],
        )
        content = response.choices[0].message.content.strip()
        logger.info(f"{self.__class__.__name__} response content: {content}")

        check_result = json.loads(content)
        return EnhancementStrategy.get_strategy(check_result["strategy"].lower())
