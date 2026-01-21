import logging
from abc import ABC, abstractmethod

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


class QueryRewriter(ABC):
    """问题重写器抽象基类"""

    @abstractmethod
    def rewrite(self, question: str) -> str:
        """重写问题"""
        raise NotImplementedError


class LLMQueryRewriter(QueryRewriter):
    """LLM问题重写器"""

    system_prompt: str = """
    <role>
    你是一位专业的客服助理,负责理解用户的真实意图,将用户的口语化、模糊或省略的问题改写为清晰、完整、便于检索的标准问题。
    </role>

    <task>
    基于用户当前问题、历史对话和产品资料片段,将用户问题改写为明确、完整的检索问题。
    </task>

    <rewrite_principles>
    1. 补全指代: 将"这个"、"它"、"那款"等指代词替换为具体的产品名称或功能
    2. 明确主语: 补充省略的主语(通常是产品名称)
    3. 扩展简略: 将"怎么样"、"如何"等笼统问法扩展为具体属性(如"性能怎么样"、"拍照效果如何")
    4. 保留意图: 保持用户原始询问的核心意图不变
    5. 标准化表达: 转换为规范的产品咨询用语,便于RAG检索
    6. 融合上下文: 结合历史对话理解当前问题的真实含义
    </rewrite_principles>

    <rewrite_rules>
    规则1: 如果当前问题包含指代词(这个、那个、它等)
    - 从历史对话中找到指代的产品或功能
    - 用具体名称替换指代词

    规则2: 如果当前问题是追问或延续性问题
    - 结合历史对话确定讨论的主题
    - 补全完整的问题表述

    规则3: 如果当前问题过于简略(单个词或短语)
    - 根据历史对话推断完整问题
    - 扩展为完整句式

    规则4: 如果产品资料片段中出现相关产品名称
    - 优先使用资料中的标准产品名称
    - 确保问题与产品资料匹配

    规则5: 如果问题已经清晰完整
    - 仅做轻微的标准化调整
    - 保持原问题的主要表述
    </rewrite_rules>

    <output_format>
    直接输出改写后的问题,不要添加任何解释或标签。
    改写后的问题应该:
    - 是一个完整的疑问句
    - 包含明确的产品名称或功能名称
    - 便于在产品知识库中检索
    - 字数控制在50字以内
    </output_format>

    <examples>
    <example>
    用户当前问题: 这个怎么样?
    历史对话: 
    用户: QuantumFlip的拍照功能如何?
    客服: QuantumFlip后置5000万AI双摄,支持双重防抖...
    产品资料片段: QuantumFlip是我们的旗舰翻盖手机...
    输出: QuantumFlip的整体表现怎么样?
    </example>

    <example>
    用户当前问题: 电池呢?
    历史对话:
    用户: VERTU手机性能如何?
    客服: VERTU搭载顶级处理器,运行流畅...
    产品资料片段: VERTU手机配备5000mAh大电池...
    输出: VERTU手机的电池续航怎么样?
    </example>

    <example>
    用户当前问题: 有什么颜色?
    历史对话:
    用户: 想看看你们的新款手机
    客服: 我们有QuantumFlip和VERTU两个系列
    产品资料片段: QuantumFlip提供曜石黑、冰川银、星云蓝...
    输出: QuantumFlip有哪些颜色可选?
    </example>

    <example>
    用户当前问题: 支持5G吗?
    历史对话: (无)
    产品资料片段: QuantumFlip全系支持5G网络...
    输出: QuantumFlip支持5G网络吗?
    </example>

    <example>
    用户当前问题: 和苹果比哪个好?
    历史对话:
    用户: VERTU耳机音质怎么样?
    客服: VERTU耳机融入伦敦交响乐团专属调校...
    产品资料片段: VERTU耳机采用Hi-Fi级解码...
    输出: VERTU耳机和苹果耳机相比哪个音质更好?
    </example>

    <example>
    用户当前问题: 多少钱?
    历史对话:
    用户: QuantumFlip拍照怎么样?
    客服: QuantumFlip后置5000万AI双摄...
    产品资料片段: QuantumFlip售价7999元起...
    输出: QuantumFlip的价格是多少?
    </example>

    <example>
    用户当前问题: VERTU手机性能如何?
    历史对话: (无)
    产品资料片段: VERTU搭载骁龙8 Gen3处理器...
    输出: VERTU手机的性能如何?
    </example>
    </examples>

    <edge_cases>
    情况1: 历史对话为空且问题已完整
    - 仅做标准化调整,补充产品名称(如有)

    情况2: 问题涉及多个产品对比
    - 明确列出对比的双方产品

    情况3: 问题过于口语化或包含语气词
    - 转换为书面化、标准化表述
    - 示例: "emmm这个能用多久啊" → "这款产品的使用寿命有多久?"

    情况4: 无法从上下文判断指代对象
    - 保留原问题,仅做语句标准化
    - 不要臆测或添加不确定的信息
    </edge_cases>

    <constraints>
    1. 改写后的问题必须忠于用户原意,不改变询问方向
    2. 不要添加用户未提及的限定条件
    3. 优先使用产品资料中的标准名称
    4. 保持问题的开放性,不要预设答案倾向
    5. 字数控制在50字以内
    </constraints>
    """

    user_prompt: str = """
    <input>
    - 当前问题: {question}
    - 历史对话: {chat_history}
    - 产品资料: {rag_context}
    </input>
    """

    def __init__(self, openai_client: AsyncOpenAI, llm_model: str):
        """初始化LLM问题重写器"""
        self.client = openai_client
        self.llm_model = llm_model

    async def rewrite(self, question: str, chat_history: str, rag_context: str) -> str:
        """重写问题"""
        response = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        question=question,
                        chat_history=chat_history,
                        rag_context=rag_context,
                    ),
                },
            ],
        )
        content = response.choices[0].message.content.strip()
        logger.info(f"{self.__class__.__name__} response content: {content}")

        return content
