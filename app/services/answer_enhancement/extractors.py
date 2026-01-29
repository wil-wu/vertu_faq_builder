import json
import logging
from typing import Any
from abc import ABC, abstractmethod

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class Extractor(ABC):
    """提取器抽象基类"""

    @abstractmethod
    def extract(self, question: str, answer: str) -> str:
        """提取答案"""
        raise NotImplementedError


class RuleExtractor(Extractor):
    """规则提取器"""

    def __init__(self, rules: list[str]):
        """初始化规则提取器"""
        self.rules = rules

    def extract(self, question: str, answer: str) -> str:
        """提取答案"""
        return answer


class MLExtractor(Extractor):
    """机器学习提取器"""

    def __init__(self, model: Any):
        """初始化机器学习提取器"""
        self.model = model

    def extract(self, question: str, answer: str) -> str:
        """提取答案"""
        return answer


class LLMExtractor(Extractor):
    """LLM提取器"""

    system_prompt: str = """
    <role>
    你是一位专业的客服助理,负责根据用户问题和优化后的答案生成图片/视频需求描述文本。
    </role>

    <task>
    根据用户问题和优化后的答案,判断是否需要图片/视频,如需要则生成描述文本用于检索资源。
    </task>

    <description_generation_rules>
    如果需要补充图片/视频,生成简洁的描述文本:
    1. 明确产品名称
    2. 指明需要的资源类型(实拍图/外观图/视频等)
    3. 可选:补充具体要求(如颜色、角度等)
    4. 字数控制在20字以内

    描述文本格式示例:
    图片类:
    - "QuantumFlip实拍样张"
    - "VERTU耳机外观图"
    - "QuantumFlip曜石黑配色图"
    - "QuantumFlip屏幕显示效果图"
    - "VERTU手机开箱配件图"
    - "QuantumFlip三色外观图"

    视频类:
    - "QuantumFlip拍照功能演示视频"
    - "VERTU手机开箱视频"
    - "QuantumFlip使用场景视频"

    如果不需要图片/视频,输出description为null。
    </description_generation_rules>

    <output_format>
    输出JSON格式:
    {
    "description": "图片/视频描述文本或null",
    "reason": "识别理由"
    }

    注意:
    - 如果需要图片/视频,description为描述文本,reason说明为什么需要
    - 如果不需要图片/视频,description为null,reason说明为什么不需要
    - reason需简洁说明(不超过30字)
    - 不要输出任何JSON之外的内容
    </output_format>

    <examples>
    <example>
    用户问题: 拍照怎么样?
    优化后答案: QuantumFlip后置5000万AI双摄,支持双重防抖。AI暗房师功能配合前置3200万镜头,自拍更立体。我给您发几张实拍图吧
    输出: {
    "description": "QuantumFlip实拍样张",
    "reason": "拍照问题需要实拍图展示效果"
    }
    </example>

    <example>
    用户问题: 有什么颜色?
    优化后答案: 提供曜石黑、冰川银、星云蓝三种配色。给您看看外观图
    输出: {
    "description": "QuantumFlip三色外观图",
    "reason": "颜色问题需要外观图展示配色"
    }
    </example>

    <example>
    用户问题: 外观设计怎么样?
    优化后答案: 采用玻璃机身,曲面屏设计,质感高端大气。给您展示一下实物
    输出: {
    "description": "QuantumFlip外观设计图",
    "reason": "外观问题需要产品图展示设计"
    }
    </example>

    <example>
    用户问题: 屏幕显示效果好吗?
    优化后答案: 采用6.7英寸AMOLED屏幕,支持120Hz刷新率,色彩鲜艳细腻。我给您发几张图片看看
    输出: {
    "description": "QuantumFlip屏幕显示效果图",
    "reason": "屏幕显示问题需要效果图展示"
    }
    </example>

    <example>
    用户问题: VERTU耳机外观如何?
    优化后答案: VERTU耳机采用金属材质,设计精致奢华。给您看看实物图
    输出: {
    "description": "VERTU耳机外观图",
    "reason": "外观问题需要产品图展示"
    }
    </example>

    <example>
    用户问题: 手机厚度怎么样?
    优化后答案: 机身厚度仅7.8mm,重量185g,手感轻薄舒适。给您看看对比图
    输出: {
    "description": "QuantumFlip厚度对比图",
    "reason": "厚度问题需要对比图直观展示"
    }
    </example>

    <example>
    用户问题: 拍照功能怎么用?
    优化后答案: 打开相机,选择AI模式即可智能识别场景。操作很简单,我给您演示一下
    输出: {
    "description": "QuantumFlip拍照功能演示视频",
    "reason": "功能演示适合用视频展示操作"
    }
    </example>

    <example>
    用户问题: 电池续航怎么样?
    优化后答案: 这款配备5000mAh大电池,支持66W快充。正常使用一整天完全没问题。
    输出: {
    "description": null,
    "reason": "续航问题无需图片,文字说明已足够"
    }
    </example>

    <example>
    用户问题: 支持5G吗?
    优化后答案: 支持5G网络,双模全网通。网速更快更稳定,体验更流畅。
    输出: {
    "description": null,
    "reason": "网络制式问题无需图片,文字说明已足够"
    }
    </example>

    <example>
    用户问题: 价格多少?
    优化后答案: 这款售价7999元起,性价比很高。
    输出: {
    "description": null,
    "reason": "价格问题无需图片,文字说明已足够"
    }
    </example>

    <example>
    用户问题: 拍照怎么样?
    优化后答案: 我给您发几张实拍图吧[QuantumFlip实拍图]
    输出: {
    "description": null,
    "reason": "答案已包含图片链接,无需重复添加"
    }
    </example>

    <example>
    用户问题: 这款手机包装怎么样?
    优化后答案: 包装采用礼盒设计,配件齐全。给您看看开箱视频
    输出: {
    "description": "QuantumFlip开箱视频",
    "reason": "开箱展示适合用视频呈现"
    }
    </example>
    </examples>
    """
    user_prompt: str = """
    <input>
    - 用户问题: {question}
    - 优化后答案: {answer}
    </input>
    """

    def __init__(
        self, openai_client: AsyncOpenAI, llm_model: str, temperature: float = 0.01
    ):
        """初始化LLM提取器"""
        self.client = openai_client
        self.llm_model = llm_model
        self.temperature = temperature

    async def extract(self, question: str, answer: str) -> str:
        """提取答案"""
        response = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        question=question, answer=answer
                    ),
                },
            ],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content.strip()
        logger.debug(f"{self.__class__.__name__} response content: {content}")

        try:
            extract_result = json.loads(content)
        except json.JSONDecodeError:
            logger.error(
                f"{self.__class__.__name__} response content is not a valid JSON: {content}"
            )
            return ""

        return extract_result.get("description", "")
