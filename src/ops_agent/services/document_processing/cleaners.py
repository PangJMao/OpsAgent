from __future__ import annotations

import re
import unicodedata

from ops_agent.services.document_processing.schema import ParsedBlock, ParsedDocument

MOJIBAKE_PATTERNS = ("锟", "�", "鐭", "绔犺", "寮曠", "鍏ュ", "閿", "瀵硅", "璇锋", "妫€")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
REPEATED_SYMBOL_RE = re.compile(r"([_\-=*#])\1{5,}")


def clean_text(text: str, *, remove_duplicate_headings: bool = True) -> str:
    """统一文本清洗：入库、检索后和最终输出前都使用，避免乱码和噪声进入生成链路。"""

    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = CONTROL_RE.sub("", text)
    text = REPEATED_SYMBOL_RE.sub("", text)
    lines = []
    previous_heading = ""
    previous_line = ""
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            continue
        if remove_duplicate_headings and line == previous_line:
            continue
        if remove_duplicate_headings and line.startswith("#") and line == previous_heading:
            continue
        if remove_duplicate_headings and previous_heading and line == previous_heading.lstrip("# ").strip():
            continue
        previous_heading = line if line.startswith("#") else previous_heading
        previous_line = line
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def mojibake_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(text.count(pattern) for pattern in MOJIBAKE_PATTERNS) / max(len(text), 1)


def is_meaningful_text(text: str) -> bool:
    cleaned = clean_text(text)
    if len(cleaned) < 2:
        return False
    if mojibake_ratio(cleaned) > 0.08:
        return False
    alnum_or_cjk = sum(1 for char in cleaned if char.isalnum() or "\u4e00" <= char <= "\u9fff")
    return alnum_or_cjk >= max(2, len(cleaned) * 0.25)


class DocumentCleaner:
    repeated_space_re = re.compile(r"[ \t]+")
    blank_line_re = re.compile(r"\n{3,}")

    def clean(self, document: ParsedDocument) -> ParsedDocument:
        cleaned_blocks = []
        seen: set[str] = set()
        for block in document.blocks:
            text = self.clean_text(block.text)
            if not text:
                continue
            fingerprint = re.sub(r"\W+", "", text.lower())[:200]
            if fingerprint and fingerprint in seen and block.block_type in {"header", "footer"}:
                continue
            seen.add(fingerprint)
            cleaned_blocks.append(
                ParsedBlock(
                    text=text,
                    block_type=block.block_type,
                    page=block.page,
                    sheet_name=block.sheet_name,
                    row_index=block.row_index,
                    slide_number=block.slide_number,
                    section=block.section,
                    heading_path=block.heading_path,
                    metadata=block.metadata,
                )
            )
        return ParsedDocument(
            source=document.source,
            file_name=document.file_name,
            file_type=document.file_type,
            title=document.title,
            parser=document.parser,
            blocks=cleaned_blocks,
            metadata=document.metadata,
        )

    def clean_text(self, text: str) -> str:
        cleaned = clean_text(text)
        return cleaned if is_meaningful_text(cleaned) else ""
