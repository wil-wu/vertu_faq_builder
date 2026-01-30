from enum import Enum


class EnhancementStrategy(Enum):
    """增强策略"""

    ENHANCE = "enhance"
    GUIDANCE = "guidance"
    DIRECT = "direct"

    @classmethod
    def get_strategy(cls, strategy: str) -> "EnhancementStrategy":
        """获取增强策略"""
        return cls(strategy.lower())

    @classmethod
    def get_strategies_values(cls) -> list[str]:
        """获取增强策略值"""
        return [strategy.value for strategy in cls]
