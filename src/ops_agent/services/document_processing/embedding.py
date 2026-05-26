from __future__ import annotations

from ops_agent.services.embedding_service import EmbeddingModel, create_embedding_model


class EmbeddingService:
    def __init__(self, model: EmbeddingModel | None = None) -> None:
        self.model = model or create_embedding_model()

    @property
    def dimensions(self) -> int:
        return self.model.dimensions

    def embed(self, text: str) -> list[float]:
        return self.model.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
