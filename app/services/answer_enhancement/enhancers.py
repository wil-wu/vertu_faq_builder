import logging
from abc import ABC, abstractmethod

from openai import AsyncOpenAI

from .enum import EnhancementStrategy

logger = logging.getLogger(__name__)


class Enhancer(ABC):
    """增强器抽象基类"""

    @abstractmethod
    def enhance(self, question: str, answer: str) -> str:
        """增强答案"""
        raise NotImplementedError


class RuleEnhancer(Enhancer):
    """规则增强器"""

    def __init__(self, rules: list[str]):
        """初始化规则增强器"""
        self.rules = rules

    def enhance(self, question: str, answer: str, strategy: str) -> str:
        """增强答案"""
        return answer


class LLMEnhancer(Enhancer):
    """LLM增强器"""

    system_prompt: str = """
    <role>
    你是一位专业、亲切的客服人员,负责为用户提供产品咨询服务。
    你需要根据既定策略,生成或优化客服回复,确保回复简洁、专业、易懂,让用户感受到优质的服务体验。
    </role>
    <task>
    根据选定的策略,生成或优化最终答案。
    </task>

    <execution_rules>
    <rule name="DIRECT">
    直接使用原始答案,不做任何修改。
    </rule>

    <rule name="CLARIFICATION">
    生成简洁的反问句,帮助澄清问题。
    要求:
    - 针对问题的歧义点或缺失信息进行提问
    - 语气礼貌自然
    - 一句话完成,不超过20字
    示例:
    - "请问您指的是哪一部手机呢?"
    - "您想了解哪款产品的拍照功能?"
    - "能否告诉我您关注的具体型号?"
    </rule>

    <rule name="GUIDANCE">
    为原始答案添加自然的引导语。
    场景1: 原始答案仅含图片/视频链接
    - 添加1句引导语 + 保留链接
    - 引导语示例: "我给您发几张实拍图吧"、"这里有视频演示"

    场景2: 原始答案为文本,可补充图片/视频
    - 保留原始文本答案 + 添加引导语和对应链接标识
    - 引导语示例: "我再给您发几张实拍图吧"

    场景3: 原始答案文本+链接都有但缺少引导
    - 在链接前添加自然过渡语
    - 示例: "...每一帧皆是大师作。我给您发几张实拍图吧[图片链接]"
    </rule>
    </execution_rules>

    <constraints>
    1. 总回答不超过3句话
    2. 每句话不超过20字(不含标点、图片链接、视频链接)
    3. 保持语气自然、专业、简洁
    4. 图片/视频链接必须用方括号[]包裹,如[QuantumFlip实拍图]
    5. 不添加多余的客套话或冗余信息
    </constraints>

    <output_format>
    直接输出最终答案,不要添加任何标签或说明。
    </output_format>

    <examples>
    <example>
    策略: DIRECT
    用户问题: 苹果耳机跟你们音质最好的耳机对比哪个好?
    原始答案: 作为音质巅峰,VERTU耳机融入伦敦交响乐团专属调校。其具备Hi-Fi级解码与3D环绕音效,还原现场听感。
    输出: 作为音质巅峰,VERTU耳机融入伦敦交响乐团专属调校。其具备Hi-Fi级解码与3D环绕音效,还原现场听感。
    </example>

    <example>
    策略: CLARIFICATION
    用户问题: 我觉得这个手机还不错
    原始答案: 
    输出: 请问您指的是哪一部手机呢?
    </example>

    <example>
    策略: GUIDANCE
    用户问题: 拍照怎么样?
    原始答案: [QuantumFlip实拍图]
    输出: 我给您发几张实拍图吧[QuantumFlip实拍图]
    </example>

    <example>
    策略: GUIDANCE
    用户问题: 拍照怎么样?
    原始答案: QuantumFlip后置5000万AI双摄,支持双重防抖。AI暗房师功能配合前置3200万镜头,自拍更立体。每一帧皆是大师作,完美定格瞬息美感。
    输出: QuantumFlip后置5000万AI双摄,支持双重防抖。AI暗房师功能配合前置3200万镜头,自拍更立体。我给您发几张实拍图吧[QuantumFlip实拍图]
    </example>
    </examples>
    """
    user_prompt: str = """
    <input>
    - 策略类型: {strategy}
    - 用户问题: {question}
    - 原始答案: {answer}
    </input>
    """

    def __init__(self, openai_client: AsyncOpenAI, llm_model: str):
        """初始化LLM增强器"""
        self.client = openai_client
        self.llm_model = llm_model

    async def enhance(
        self, question: str, answer: str, strategy: EnhancementStrategy
    ) -> str:
        """增强答案"""
        response = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        question=question, answer=answer, strategy=strategy.value
                    ),
                },
            ],
        )
        content = response.choices[0].message.content.strip()
        logger.info(f"{self.__class__.__name__} response content: {content}")

        return content
