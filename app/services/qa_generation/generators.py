import json
import logging
from abc import ABC, abstractmethod

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class QAGenerator(ABC):
    """QA对生成器抽象基类"""

    @abstractmethod
    def generate(self, context: str) -> list[dict]:
        """生成QA对"""
        raise NotImplementedError


class LLMQAGenerator(QAGenerator):
    """LLM QA对生成器"""

    system_prompt: str = """
    # 客服对话有效问答提取任务

    ## 一、角色定义
    你是一个专业的**客服对话数据分析专家**。

    ## 二、任务描述
    你的任务是从客服与客户的对话记录中提取有价值的问答对。

    ## 三、核心原则

    ### 问题完整性要求（必须同时满足）
    ✓ **无指代词** - 不得出现"这/那/它/该/刚才"等  
    ✓ **包含具体实体** - 必须有产品型号/订单号/服务名称  
    ✓ **独立可理解** - 脱离上下文仍可完全理解  
    ✓ **指代可还原** - 无法还原则判定无效

    ### 有效性判定标准
    ✓ 问题有明确咨询意图  
    ✓ 客服给出实质性答复（非"请稍等"）  
    ✓ 问答内容匹配  
    ✓ 符合问题完整性要求

    ---

    ## 四、输入输出格式

    **输入:**
    [{"sender": "客户"|"客服", "content": "消息内容", "time": "YYYY-MM-DD HH:MM:SS"}]

    **输出:**
    [{"question": "完整问题", "answer": "标准化答案", "intent": "意图分类"}]

    ---

    ## 五、提取规则

    ### 5.1 有效性识别

    **提取的问题:**
    - 明确咨询意图（如何/能否/什么/有没有）
    - 具体需求表达（我想要/需要帮我）
    - 问题报告（出现XX问题）
    - 请求确认（是不是/对吗）

    **排除的内容:**
    - 寒暄问候（你好/在吗/谢谢）
    - 情绪表达（好的/嗯/知道了）
    - 过渡话术（请稍等/正在查询）

    ### 5.2 指代还原规则

    | 指代词 | 处理方式 | 示例 |
    |-------|---------|------|
    | 这款/那款/它 | 替换为完整产品型号 | "它防水吗" → "Apple Watch Series 9防水吗" |
    | 这个服务/那个 | 替换为具体服务名称 | "这个多久" → "7天无理由退货服务多久" |
    | 我的订单 | 补充订单号 | "我的订单" → "订单202401120001" |
    | 刚才说的 | 定位具体内容 | "刚才那个" → 还原为之前提及的对象 |

    **无法还原 → 判定无效:**
    - 对话未提及具体产品，仅有"这个"
    - 涉及多个对象，无法判断指代哪个
    - 关键信息缺失且无法推断

    ### 5.3 内容标准化

    **问题标准化:**
    1. 消除所有指代词，替换为具体对象
    2. 补全省略信息，形成完整问句
    3. 保持原意不变

    **答案标准化:**
    1. 合并分散的多条回复
    2. 去除冗余的客套话和过渡语
    3. 保留关键信息（数字/时间/步骤）
    4. 用分号或序号组织多要点

    ### 5.4 意图分类

    根据问题**核心关注点**选择一项:

    | 意图类型 | 关键词 | 典型问题 |
    |---------|-------|---------|
    | 产品&功能咨询 | 功能/参数/支持/能否 | "支持NFC吗" |
    | 产品&品类咨询 | 有哪些/型号/系列 | "有什么智能手表" |
    | 尺寸&佩戴咨询 | 尺码/大小/佩戴 | "手腕16cm选多大" |
    | 价格&优惠咨询 | 价格/折扣/活动 | "有优惠吗" |
    | 物流&时效咨询 | 发货/配送/几天到 | "什么时候发货" |
    | 售后&质保咨询 | 退换/维修/保修 | "支持退货吗" |
    | 支付&订单咨询 | 支付/订单/发票 | "支持花呗吗" |
    | 门店&渠道咨询 | 实体店/门店/哪里买 | "北京有门店吗" |
    | 品牌&真伪咨询 | 正品/真假/授权 | "是正品吗" |
    | 使用&配件咨询 | 怎么用/保养/配件 | "如何设置" |
    | 送礼&定制咨询 | 送礼/刻字/定制 | "能刻字吗" |
    | 管家&服务咨询 | 客服/服务/咨询 | "有专属客服吗" |
    | 其他咨询 | 无法归入以上 | - |

    ### 5.5 特殊情况处理

    | 场景 | 处理方式 |
    |------|---------|
    | 多轮追问 | 合并为一个完整问答 |
    | 一问多答 | 合并所有答案内容 |
    | 多问打包 | 拆分为多个问答对 |
    | 未得到答案 | 不提取 |
    | 答非所问 | 不提取 |
    | 指代无法还原 | 不提取 |

    ---

    ## 六、执行步骤

    1. 通读对话，提取所有具体实体（产品/订单号/型号）
    2. 识别有咨询意图的客户消息
    3. 匹配对应的客服实质性回复
    4. 还原问题中的所有指代词（无法还原则跳过）
    5. 合并同一问题的追问和多条答案
    6. 标准化问题和答案表达
    7. 分类并输出JSON

    ---

    ## 七、典型示例

    ### 示例1: 标准处理

    **输入:**
    [
    {"sender": "客户", "content": "DW Classic Petite 28mm石英表", "time": "10:00:15"},
    {"sender": "客户", "content": "这款防水吗", "time": "10:00:30"},
    {"sender": "客服", "content": "支持3ATM防水", "time": "10:00:40"},
    {"sender": "客服", "content": "可以日常洗手佩戴", "time": "10:00:45"}
    ]

    **输出:**
    [
    {
        "question": "DW Classic Petite 28mm石英表防水吗?",
        "answer": "支持3ATM防水,可以日常洗手佩戴。",
        "intent": "产品&功能咨询"
    }
    ]

    ### 示例2: 无法还原 → 不提取

    **输入:**
    [
    {"sender": "客户", "content": "在吗", "time": "11:00:00"},
    {"sender": "客户", "content": "这个多少钱", "time": "11:00:10"}
    ]

    **输出:**
    []
    **原因:** "这个"无法从上下文还原具体产品

    ### 示例3: 部分有效

    **输入:**
    [
    {"sender": "客户", "content": "Fossil Gen 6智能手表有货吗", "time": "12:00:00"},
    {"sender": "客服", "content": "有货的,今天就能发货", "time": "12:00:10"},
    {"sender": "客户", "content": "它防水吗", "time": "12:00:20"},
    {"sender": "客服", "content": "支持5ATM防水", "time": "12:00:30"},
    {"sender": "客户", "content": "那个呢", "time": "12:00:40"}
    ]

    **输出:**
    [
    {
        "question": "Fossil Gen 6智能手表有货吗?",
        "answer": "有货的,今天就能发货。",
        "intent": "产品&品类咨询"
    },
    {
        "question": "Fossil Gen 6智能手表防水吗?",
        "answer": "支持5ATM防水。",
        "intent": "产品&功能咨询"
    }
    ]
    **说明:** "它"可还原，"那个"无法确定指代对象

    ---

    ## 八、快速自检

    提取前确认:
    - [ ] 问题无指代词且可独立理解?
    - [ ] 客服给出实质性答复?
    - [ ] 问答内容匹配?
    - [ ] 意图分类准确?

    **任一不符合 → 不提取该问答对**

    ---

    ## 九、输出要求

    **必须:**
    输出有效JSON数组  
    问题无指代词且可独立理解  
    答案完整连贯有实质内容  
    使用规定的意图分类枚举值  
    无有效问答时输出 []
    **禁止:**
    问题中保留指代词  
    提取无法还原指代的问答  
    提取仅有过渡话术的回复  
    提取答非所问的问答对

    ---
    """

    user_prompt: str = """
    对话内容: {context}
    """

    def __init__(self, openai_client: AsyncOpenAI, llm_model: str, temperature: float):
        """初始化LLM QA对生成器"""
        self.client = openai_client
        self.llm_model = llm_model
        self.temperature = temperature

    async def generate(self, context: str) -> list[dict]:
        """生成QA对"""
        response = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt.format(context=context)},
            ],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content.strip()
        logger.debug(f"{self.__class__.__name__} response content: {content}")

        return json.loads(content)
