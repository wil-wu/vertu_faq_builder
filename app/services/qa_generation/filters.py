import re
import json
import logging
from abc import ABC, abstractmethod

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class Filter(ABC):
    """过滤器抽象基类"""

    @abstractmethod
    def filter(self, qa_pair: dict) -> bool:
        """过滤QA对"""
        raise NotImplementedError


class RuleFilter(Filter):
    """规则过滤器"""

    def __init__(self, rules: list[dict]):
        """初始化规则过滤器"""
        self.rules = rules

    async def filter(self, qa_pair: dict) -> bool:
        """过滤QA对"""
        for rule in self.rules:
            if rule.get("question_condition") and not re.search(
                rule.get("question_condition"), qa_pair["question"], re.I
            ):
                return False
            if rule.get("answer_condition") and not re.search(
                rule.get("answer_condition"), qa_pair["answer"], re.I
            ):
                return False
            if rule.get("intent_condition") and not re.search(
                rule.get("intent_condition"), qa_pair["intent"], re.I
            ):
                return False
        return True


class LLMFilter(Filter):
    """LLM过滤器"""

    system_prompt: str = """
    <role>
    你是一个产品相关性判断专家。
    </role>

    <task>
    你的任务是判断给定的问答对是否与提供的产品型号列表相关。
    </task>

    <input_format>
    你将收到以下输入:
    1. 产品型号列表:一个包含多个产品型号的列表
    2. 问答对:格式为 {"question": "xx", "answer": "xx", "intent": "xx"}
    </input_format>

    <判断标准>
    问答对与产品列表"相关"的情况包括:
    - 问题或答案中直接提到了列表中的产品型号(完全匹配或部分匹配)
    - 问题或答案描述的功能、特性、问题明确对应列表中的某个产品
    - 问题或答案中提到的产品系列、产品线属于列表中的产品范围
    - 问答内容是关于列表中产品的使用、故障、配置、参数等

    问答对与产品列表"不相关"的情况包括:
    - 提到的产品型号完全不在列表中
    - 是关于其他品牌或完全不同类型的产品
    - 是通用性问题,没有特定产品指向
    - 产品型号相似但明确是不同型号(需仔细比对)
    </判断标准>

    <reasoning_process>
    在做出判断前,请按以下步骤思考:
    1. 提取问答对中提到的所有产品型号、产品名称或产品特征
    2. 将提取的信息与产品列表逐一比对
    3. 考虑可能的变体(如型号缩写、系列名称等)
    4. 基于比对结果做出判断
    5. 简洁说明判断理由
    </reasoning_process>

    <output_format>
    请严格按照以下JSON格式输出,不要包含其他内容:
    {
    "keep": true/false,
    "reason": "简洁说明判断理由,指出匹配的产品型号或不匹配的原因"
    }

    示例:
    - 如果相关: {"keep": true, "reason": "问答提到产品型号X100,在列表中存在"}
    - 如果不相关: {"keep": false, "reason": "问答提到产品型号Y200,不在提供的产品列表中"}
    </output_format>

    <注意事项>
    - 对边界情况保持谨慎:如果不确定,倾向于保留(keep: true)
    - 注意产品型号的变体和简称
    - 理由需要具体,指出关键匹配点或不匹配点
    - 严格输出JSON格式,确保可被程序解析
    </注意事项>

    <产品型号列表>
    VERTU AGENT Q 手机
    </产品型号列表>
    """

    user_prompt: str = """
    <input>
    - 问答对: {qa_pair}
    </input>
    """

    def __init__(self, openai_client: AsyncOpenAI, llm_model: str, temperature: float):
        """初始化LLM过滤器"""
        self.client = openai_client
        self.llm_model = llm_model
        self.temperature = temperature

    async def filter(self, qa_pair: dict) -> bool:
        """过滤QA对"""
        response = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt.format(qa_pair=qa_pair)},
            ],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content.strip()
        logger.debug(f"{self.__class__.__name__} response content: {content}")

        try:
            filter_result = json.loads(content)
        except json.JSONDecodeError:
            logger.error(
                f"{self.__class__.__name__} response content is not a valid JSON: {content}"
            )
            return True

        return filter_result.get("keep", True)
