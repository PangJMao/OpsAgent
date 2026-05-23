from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from typing import Protocol

from ops_agent.config import settings

TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class EmbeddingModel(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]:
        ...


class HashingEmbeddingModel:
    """Deterministic local fallback used when no real embedding endpoint is configured."""

    def __init__(self, dimensions: int = settings.embedding_dimensions) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        return _normalize(vector)

    def _tokens(self, text: str) -> list[str]:
        tokens = [token.lower() for token in TOKEN_RE.findall(text)]
        cjk_chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
        tokens.extend(a + b for a, b in zip(cjk_chars, cjk_chars[1:]))
        return tokens


class OpenAICompatibleEmbeddingModel:
    """Real embedding model through an OpenAI-compatible /embeddings endpoint."""

    def __init__(
        self,
        api_key: str = settings.embedding_api_key,
        base_url: str = settings.embedding_base_url,
        model: str = settings.embedding_model,
        dimensions: int = settings.embedding_dimensions,
        timeout_seconds: float = settings.embedding_timeout_seconds,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def embed(self, text: str) -> list[float]:
        if not self.enabled:
            raise RuntimeError("Embedding endpoint is not configured.")

        payload: dict[str, object] = {"model": self.model, "input": text}
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc

        data = json.loads(body)
        records = data.get("data") or []
        if not records:
            raise RuntimeError("Embedding endpoint returned no vectors.")
        vector = records[0].get("embedding")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError("Embedding endpoint returned an invalid vector.")
        numeric_vector = [float(value) for value in vector]
        if self.dimensions and len(numeric_vector) != self.dimensions:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {self.dimensions}, got {len(numeric_vector)}."
            )
        return _normalize(numeric_vector)


class ResilientEmbeddingModel:
    def __init__(
        self,
        primary: EmbeddingModel,
        fallback: EmbeddingModel | None = None,
        require_primary: bool = False,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.require_primary = require_primary
        self.dimensions = primary.dimensions

    def embed(self, text: str) -> list[float]:
        try:
            return self.primary.embed(text)
        except RuntimeError:
            if self.require_primary or self.fallback is None:
                raise
            return self.fallback.embed(text)


def create_embedding_model() -> EmbeddingModel:
    provider = settings.embedding_provider.lower()
    if provider in {"openai", "openai-compatible", "remote", "real"}:
        return ResilientEmbeddingModel(
            primary=OpenAICompatibleEmbeddingModel(),
            fallback=HashingEmbeddingModel(),
            require_primary=settings.require_external_services,
        )
    return HashingEmbeddingModel()


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
