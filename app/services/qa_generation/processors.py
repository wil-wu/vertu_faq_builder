import logging
from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Processor(ABC):
    """处理器抽象基类"""

    @abstractmethod
    def process(self, qas: list[dict]) -> list[dict]:
        """处理QA对"""
        raise NotImplementedError


class SemanticProcessor(Processor):
    """语义处理器"""

    def __init__(
        self, sentence_transformer: SentenceTransformer, semantic_threshold: float
    ):
        self.sentence_transformer = sentence_transformer
        self.semantic_threshold = semantic_threshold

    async def process(self, qas: list[dict]) -> list[dict]:
        """处理QA对，移除相似度大于阈值的重复问答对"""
        if not qas:
            return qas

        # 提取所有问题
        questions = [qa["question"] for qa in qas]

        # 对问题进行编码
        embeddings = self.sentence_transformer.encode(questions, convert_to_numpy=True)

        # 用于存储需要保留的索引
        keep_indices = []

        removed_qas = []

        # 遍历每个问答对
        for i in range(len(qas)):
            should_keep = True

            # 与之前保留的问答对比较相似度
            for j in keep_indices:
                # 计算余弦相似度
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )

                # 如果相似度大于阈值，则跳过当前问答对
                if similarity > self.semantic_threshold:
                    should_keep = False
                    removed_qas.append(qas[i])
                    break

            # 如果应该保留，添加到保留列表
            if should_keep:
                keep_indices.append(i)

        logger.debug(f"{self.__class__.__name__} removed qas: {removed_qas}")
        # 返回保留的问答对
        return [qas[i] for i in keep_indices]
