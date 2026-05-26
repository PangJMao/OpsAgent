from __future__ import annotations

from pathlib import Path

from ops_agent.services.document_processing.parsers import (
    CsvParser,
    DocxParser,
    DocumentParser,
    LegacyDocParser,
    HtmlDocumentParser,
    ImageOcrParser,
    MarkdownParser,
    PdfParser,
    PptxParser,
    TextParser,
    XlsxParser,
)
from ops_agent.services.document_processing.schema import ParsedDocument


class DocumentLoader:
    def __init__(self, parsers: dict[str, DocumentParser] | None = None) -> None:
        self.parsers = parsers or {
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
            ".txt": TextParser(),
            ".pdf": PdfParser(),
            ".doc": LegacyDocParser(),
            ".docx": DocxParser(),
            ".xlsx": XlsxParser(),
            ".csv": CsvParser(),
            ".pptx": PptxParser(),
            ".html": HtmlDocumentParser(),
            ".htm": HtmlDocumentParser(),
            ".png": ImageOcrParser(),
            ".jpg": ImageOcrParser(),
            ".jpeg": ImageOcrParser(),
            ".tif": ImageOcrParser(),
            ".tiff": ImageOcrParser(),
        }

    @property
    def supported_suffixes(self) -> set[str]:
        return set(self.parsers)

    def load(self, path: Path) -> ParsedDocument:
        suffix = path.suffix.lower()
        parser = self.parsers.get(suffix)
        if parser is None:
            raise ValueError(f"Unsupported document type: {suffix}")
        return parser.parse(path)

    def iter_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() in self.parsers else []
        return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in self.parsers)
