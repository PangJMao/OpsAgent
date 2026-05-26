from __future__ import annotations

import re

from ops_agent.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from ops_agent.services.document_processing.cleaners import clean_text
from ops_agent.services.retriever.schema import HybridCandidate


class Deduplicator:
    def __init__(self, config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG) -> None:
        self.config = config

    def deduplicate(self, candidates: list[HybridCandidate]) -> list[HybridCandidate]:
        seen_chunks: set[str] = set()
        seen_text: set[str] = set()
        sheet_counts: dict[str, int] = {}
        result: list[HybridCandidate] = []
        for candidate in sorted(candidates, key=lambda item: item.rerank_score or item.hybrid_score, reverse=True):
            chunk_id = candidate.hit.chunk.chunk_id
            text_key = _fingerprint(candidate.hit.chunk.text)
            sheet_key = _sheet_key(candidate)
            if chunk_id in seen_chunks or text_key in seen_text:
                continue
            if sheet_counts.get(sheet_key, 0) >= self.config.max_per_sheet:
                continue
            seen_chunks.add(chunk_id)
            seen_text.add(text_key)
            sheet_counts[sheet_key] = sheet_counts.get(sheet_key, 0) + 1
            result.append(candidate)
        return result


def _fingerprint(text: str) -> str:
    return re.sub(r"\W+", "", clean_text(text).lower())[:180]


def _sheet_key(candidate: HybridCandidate) -> str:
    metadata = candidate.hit.chunk.metadata
    return f"{metadata.get('doc_name') or candidate.hit.chunk.title}|{metadata.get('sheet_name') or ''}"
