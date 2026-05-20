from __future__ import annotations

import hashlib
import math
import re

from ops_agent.core.config import settings

TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class HashingEmbeddingModel:
    """确定性的本地 embedding 基线实现。

    第一阶段先保持实现简单且可观测。后续真实 embedding 服务只需要实现同样的
    `embed` 方法即可替换。
    """

    def __init__(self, dimensions: int = settings.embedding_dimensions) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> list[str]:
        tokens = [token.lower() for token in TOKEN_RE.findall(text)]
        # 在正式引入中文分词器前，用中文字符 bigram 提升基础匹配效果。
        cjk_chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
        tokens.extend(a + b for a, b in zip(cjk_chars, cjk_chars[1:]))
        return tokens


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))
