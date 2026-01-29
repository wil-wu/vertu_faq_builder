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
        return EnhancementStrategy.DIRECT


class MLChecker(Checker):
    """机器学习模型检查器"""

    def __init__(self, model: Any):
        """初始化机器学习模型检查器"""
        self.model = model

    def check(self, question: str, answer: str) -> EnhancementStrategy:
        """策略判断"""
        return EnhancementStrategy.DIRECT


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
    1. DIRECT: 原始答案完整、优秀,可直接使用
    2. GUIDANCE: 需要为图片/视频添加引导语(原始答案包含链接但缺引导语,或不包含链接但适合添加)
    3. ENHANCE: 原始答案内容正确但表达不够优秀,需要优化改写
    </strategies>

    <analysis_steps>
    1. 检查原始答案中是否包含图片/视频链接
    - 如包含链接且缺少引导语 → GUIDANCE
    
    2. 检查问题类型是否适合补充图片/视频
    - 如果是拍照、外观、颜色、设计、屏幕、尺寸等视觉类问题 → GUIDANCE
    
    3. 检查原始答案的表达质量
    - 如果表达生硬、啰嗦、不够客服化、缺乏亲和力 → ENHANCE
    - 如果语句不通顺、结构混乱、专业术语过多 → ENHANCE
    - 如果回答过于简略,需要更完整的表达 → ENHANCE
    
    4. 其他情况(答案完整且表达优秀) → DIRECT
    </analysis_steps>

    <output_format>
    输出JSON格式,包含策略名称和决策理由:
    {
    "strategy": "策略名称",
    "reason": "简要决策原因"
    }

    注意:
    - strategy值只能是: DIRECT, GUIDANCE, ENHANCE
    - reason需简洁说明选择该策略的关键原因(不超过30字)
    - 不要输出任何JSON之外的内容
    </output_format>

    <examples>
    <example>
    用户问题: 苹果耳机跟你们音质最好的耳机对比哪个好?
    原始答案: 作为音质巅峰,VERTU耳机融入伦敦交响乐团专属调校。其具备Hi-Fi级解码与3D环绕音效,还原现场听感。
    输出: {
    "strategy": "DIRECT",
    "reason": "答案完整且表达专业优秀"
    }
    </example>

    <example>
    用户问题: 拍照怎么样?
    原始答案: [QuantumFlip实拍图]
    输出: {
    "strategy": "GUIDANCE",
    "reason": "包含图片链接但缺少引导语"
    }
    </example>

    <example>
    用户问题: 拍照怎么样?
    原始答案: QuantumFlip后置5000万AI双摄,支持双重防抖。[QuantumFlip实拍图]
    输出: {
    "strategy": "GUIDANCE",
    "reason": "包含图片链接但缺少引导语"
    }
    </example>

    <example>
    用户问题: 拍照怎么样?
    原始答案: QuantumFlip后置5000万AI双摄,支持双重防抖。AI暗房师功能配合前置3200万镜头,自拍更立体。
    输出: {
    "strategy": "GUIDANCE",
    "reason": "拍照类问题适合补充实拍图"
    }
    </example>

    <example>
    用户问题: 有什么颜色?
    原始答案: 提供曜石黑、冰川银、星云蓝三种配色
    输出: {
    "strategy": "GUIDANCE",
    "reason": "颜色类问题适合补充产品图"
    }
    </example>

    <example>
    用户问题: 外观设计怎么样?
    原始答案: 采用玻璃机身,曲面屏设计,质感高端大气。
    输出: {
    "strategy": "GUIDANCE",
    "reason": "外观类问题适合补充产品图"
    }
    </example>

    <example>
    用户问题: 电池续航怎么样?
    原始答案: 本产品配备5000mAh电池容量,支持66W快充技术,正常使用情况下可以使用一天。
    输出: {
    "strategy": "ENHANCE",
    "reason": "表达较生硬,缺乏客服亲和力,需优化"
    }
    </example>

    <example>
    用户问题: 支持5G吗?
    原始答案: 支持5G网络,包括SA和NSA双模组网方式,支持N1/N3/N28A/N41/N77/N78/N79等频段。
    输出: {
    "strategy": "ENHANCE",
    "reason": "专业术语过多,需要更客服化的表达"
    }
    </example>

    <example>
    用户问题: 价格多少?
    原始答案: 7999元起
    输出: {
    "strategy": "ENHANCE",
    "reason": "回答过于简略,需要更完整的表达"
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

    def __init__(
        self, openai_client: AsyncOpenAI, llm_model: str, temperature: float = 0.01
    ):
        """初始化LLM检查器"""
        self.client = openai_client
        self.llm_model = llm_model
        self.temperature = temperature

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
            temperature=self.temperature,
        )
        content = response.choices[0].message.content.strip()
        logger.debug(f"{self.__class__.__name__} response content: {content}")

        try:
            check_result = json.loads(content)
        except json.JSONDecodeError:
            logger.error(
                f"{self.__class__.__name__} response content is not a valid JSON: {content}"
            )
            return EnhancementStrategy.DIRECT

        return EnhancementStrategy.get_strategy(
            check_result.get("strategy", "direct").lower()
        )
