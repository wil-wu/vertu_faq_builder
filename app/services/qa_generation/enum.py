from enum import Enum


class Intent(Enum):
    """意图分类"""

    PRODUCT_FUNCTION = "产品&功能咨询"
    PRODUCT_CATEGORY = "产品&品类咨询"
    SIZE_WEAR = "尺寸&佩戴咨询"
    HOUSEKEEPING_SERVICE = "管家&服务咨询"
    PRICE_DISCOUNT = "价格&优惠咨询"
    STORE_CHANNEL = "门店&渠道咨询"
    BRAND_AUTH = "品牌&真伪咨询"
    USE_ACCESSORY = "使用&配件咨询"
    AFTER_SALE_SERVICE = "售后&质保咨询"
    GIFT_CUSTOMIZATION = "送礼&定制咨询"
    LOGISTICS_TIME = "物流&时效咨询"
    PAYMENT_ORDER = "支付&订单咨询"
    OTHER = "其他咨询"

    @classmethod
    def get_intent(cls, intent: str) -> "Intent":
        """获取意图分类"""
        return cls(intent.lower())

    @classmethod
    def get_intents_values(cls) -> list[str]:
        """获取意图分类值"""
        return [intent.value for intent in cls]


class ProductType(Enum):
    """产品类型"""

    IVERTU = "IVERTU"
    METAVERTU = "METAVERTU"
    METAVERTU_2 = "METAVERTU 2"
    SIGNATURE_4G = "Signature 4G"
    SIGNATURE_S = "Signature S"
    VERTU_AGENT_Q = "VERTU AGENT Q"
    VERTU_AGENT_IRONFLIP = "VERTU AGENT IRONFLIP"
    VERTU_QUANTUM = "VERTU QUANTUM"

    @classmethod
    def get_product_type(cls, product_type: str) -> "ProductType":
        """获取产品类型"""
        return cls(product_type.lower())

    @classmethod
    def get_product_types_values(cls) -> list[str]:
        """获取产品类型值"""
        return [product_type.value for product_type in cls]
