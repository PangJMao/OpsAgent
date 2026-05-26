from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from ops_agent.config import settings
from ops_agent.models import Chunk, Document, NormalizedDocument

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".doc", ".docx", ".xlsx", ".xls", ".csv"}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class MarkdownSection:
    text: str
    start_char: int
    end_char: int
    heading_path: list[str]
    heading_level: int


def normalize_to_markdown(path: Path) -> NormalizedDocument:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return _normalize_markdown(path)
    if suffix == ".txt":
        return _normalize_text(path)
    if suffix == ".pdf":
        return _normalize_pdf(path)
    if suffix == ".doc":
        return _normalize_with_document_processing(path)
    if suffix == ".docx":
        return _normalize_docx(path)
    if suffix == ".xlsx":
        return _normalize_xlsx(path)
    if suffix == ".xls":
        return _normalize_xls(path)
    if suffix == ".csv":
        return _normalize_csv(path)
    raise ValueError(f"暂不支持归一化该文档类型：{suffix}")


def load_text_document(path: Path) -> Document:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"不支持的文档类型：{path.suffix!r}。当前支持：{supported}")

    normalized = normalize_to_markdown(path)
    document_id = hashlib.sha256(f"{path.name}:{normalized.markdown}".encode("utf-8")).hexdigest()[:16]
    return Document(
        document_id=document_id,
        title=normalized.title,
        source_path=str(path),
        content=normalized.markdown,
        metadata={
            **normalized.metadata,
            "source": str(path),
            "source_path": str(path),
            "source_suffix": path.suffix.lower(),
            "normalized_format": "markdown",
            "bytes": path.stat().st_size,
        },
    )


def _normalize_markdown(path: Path) -> NormalizedDocument:
    markdown = path.read_text(encoding="utf-8").strip()
    return NormalizedDocument(
        title=_first_heading(markdown) or path.stem,
        markdown=markdown,
        metadata={"source_format": "markdown", "normalizer": "markdown_passthrough"},
    )


def _normalize_text(path: Path) -> NormalizedDocument:
    text = path.read_text(encoding="utf-8").strip()
    title = path.stem
    markdown = f"# {title}\n\n{text}" if text else f"# {title}"
    return NormalizedDocument(
        title=title,
        markdown=markdown,
        metadata={"source_format": "text", "normalizer": "text_to_markdown"},
    )


def _normalize_pdf(path: Path) -> NormalizedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("解析 PDF 需要安装依赖：pypdf。") from exc

    reader = PdfReader(str(path))
    page_texts = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            page_texts.append(f"## Page {index}\n\n{text}")

    markdown = f"# {path.stem}\n\n" + "\n\n".join(page_texts)
    return NormalizedDocument(
        title=path.stem,
        markdown=markdown.strip(),
        metadata={
            "source_format": "pdf",
            "normalizer": "pdf_to_markdown",
            "page_count": len(reader.pages),
        },
    )


def _normalize_docx(path: Path) -> NormalizedDocument:
    if not zipfile.is_zipfile(path):
        return _normalize_with_document_processing(path)

    paragraphs: list[str] = []
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml)
    namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for paragraph in root.findall(".//w:p", namespaces):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespaces)]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)

    markdown = f"# {path.stem}\n\n" + "\n\n".join(paragraphs)
    return NormalizedDocument(
        title=path.stem,
        markdown=markdown.strip(),
        metadata={
            "source_format": "docx",
            "normalizer": "docx_to_markdown",
            "paragraph_count": len(paragraphs),
        },
    )


def _normalize_xlsx(path: Path) -> NormalizedDocument:
    shared_strings: list[str] = []
    sheets: list[tuple[str, str]] = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "xl/sharedStrings.xml" in names:
            shared_strings = _read_xlsx_shared_strings(archive.read("xl/sharedStrings.xml"))
        workbook_names = _read_xlsx_sheet_names(archive.read("xl/workbook.xml")) if "xl/workbook.xml" in names else {}
        for name in sorted(names):
            if not re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name):
                continue
            sheet_number = re.search(r"sheet(\d+)\.xml", name)
            default_title = f"Sheet {sheet_number.group(1)}" if sheet_number else Path(name).stem
            title = workbook_names.get(default_title.replace("Sheet ", ""), default_title)
            table = _read_xlsx_sheet(archive.read(name), shared_strings)
            if table:
                sheets.append((title, table))

    sections = [f"## {title}\n\n{table}" for title, table in sheets]
    markdown = f"# {path.stem}\n\n" + "\n\n".join(sections)
    return NormalizedDocument(
        title=path.stem,
        markdown=markdown.strip(),
        metadata={
            "source_format": "xlsx",
            "normalizer": "xlsx_to_markdown",
            "sheet_count": len(sheets),
        },
    )


def _normalize_xls(path: Path) -> NormalizedDocument:
    try:
        import xlrd
    except ImportError as exc:
        raise RuntimeError("解析 .xls 需要安装依赖：xlrd。") from exc

    workbook = xlrd.open_workbook(str(path))
    sections = []
    for sheet in workbook.sheets():
        rows = []
        for row_index in range(sheet.nrows):
            values = [str(sheet.cell_value(row_index, col_index)).strip() for col_index in range(sheet.ncols)]
            values = [value for value in values if value]
            if values:
                rows.append(" | ".join(values))
        if rows:
            sections.append(f"## {sheet.name}\n\n" + "\n".join(rows))

    markdown = f"# {path.stem}\n\n" + "\n\n".join(sections)
    return NormalizedDocument(
        title=path.stem,
        markdown=markdown.strip(),
        metadata={
            "source_format": "xls",
            "normalizer": "xls_to_markdown",
            "sheet_count": len(sections),
        },
    )


def _normalize_csv(path: Path) -> NormalizedDocument:
    return _normalize_with_document_processing(path)


def _normalize_with_document_processing(path: Path) -> NormalizedDocument:
    from ops_agent.services.document_processing.service import DocumentProcessingService

    return DocumentProcessingService().to_markdown(path)


def _read_xlsx_shared_strings(xml: bytes) -> list[str]:
    root = ElementTree.fromstring(xml)
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: list[str] = []
    for item in root.findall(".//x:si", namespace):
        text = "".join(node.text or "" for node in item.findall(".//x:t", namespace)).strip()
        strings.append(text)
    return strings


def _read_xlsx_sheet_names(xml: bytes) -> dict[str, str]:
    root = ElementTree.fromstring(xml)
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    names: dict[str, str] = {}
    for index, sheet in enumerate(root.findall(".//x:sheet", namespace), start=1):
        name = sheet.attrib.get("name")
        if name:
            names[str(index)] = name
    return names


def _read_xlsx_sheet(xml: bytes, shared_strings: list[str]) -> str:
    root = ElementTree.fromstring(xml)
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: list[str] = []
    for row in root.findall(".//x:sheetData/x:row", namespace):
        values = []
        for cell in row.findall("x:c", namespace):
            values.append(_read_xlsx_cell(cell, shared_strings, namespace))
        values = [value for value in values if value]
        if values:
            rows.append(" | ".join(values))
    return "\n".join(rows)


def _read_xlsx_cell(
    cell: ElementTree.Element,
    shared_strings: list[str],
    namespace: dict[str, str],
) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//x:t", namespace)).strip()

    value_node = cell.find("x:v", namespace)
    if value_node is None or value_node.text is None:
        return ""

    value = value_node.text.strip()
    if cell_type == "s":
        try:
            return shared_strings[int(value)]
        except (IndexError, ValueError):
            return value
    return value


def persist_source_document(path: Path, document_id: str) -> Path:
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    target = settings.documents_dir / f"{document_id}{path.suffix.lower()}"
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
    return target


def persist_normalized_markdown(document: Document) -> Path:
    settings.normalized_dir.mkdir(parents=True, exist_ok=True)
    target = settings.normalized_dir / f"{document.document_id}.md"
    target.write_text(document.content, encoding="utf-8")
    return target


def chunk_document(document: Document) -> list[Chunk]:
    """优先按 Markdown 标题切分，标题缺失或章节过长时用重叠窗口兜底。"""

    markdown = document.content.strip()
    if not markdown:
        return []

    sections = _split_markdown_sections(markdown)
    if not sections:
        return _window_chunks(document, markdown, 0, [], 0, "overlap_window_fallback", True)

    chunks: list[Chunk] = []
    for section in sections:
        if len(section.text) <= settings.chunk_size:
            chunks.append(
                _build_chunk(
                    document=document,
                    text=_with_heading_context(section.text, section.heading_path),
                    start_char=section.start_char,
                    end_char=section.end_char,
                    heading_path=section.heading_path,
                    heading_level=section.heading_level,
                    strategy="markdown_heading",
                    fallback_used=False,
                    chunk_index=len(chunks),
                )
            )
            continue

        chunks.extend(
            _window_chunks(
                document=document,
                text=section.text,
                base_start=section.start_char,
                heading_path=section.heading_path,
                heading_level=section.heading_level,
                strategy="markdown_heading_window_fallback",
                fallback_used=True,
                start_index=len(chunks),
            )
        )

    return chunks


def _split_markdown_sections(markdown: str) -> list[MarkdownSection]:
    matches = list(HEADING_RE.finditer(markdown))
    if not matches:
        return []

    sections: list[MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        heading_stack = [(item_level, item_title) for item_level, item_title in heading_stack if item_level < level]
        heading_stack.append((level, title))
        heading_path = [item_title for _, item_title in heading_stack]
        section_text = markdown[start:end].strip()
        if section_text:
            sections.append(MarkdownSection(section_text, start, end, heading_path, level))
    return sections


def _window_chunks(
    document: Document,
    text: str,
    base_start: int,
    heading_path: list[str],
    heading_level: int,
    strategy: str,
    fallback_used: bool,
    start_index: int = 0,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + settings.chunk_size, len(text))
        candidate = text[start:end]
        boundary = max(candidate.rfind("\n\n"), candidate.rfind("。"), candidate.rfind("."))
        if boundary > settings.chunk_size * 0.55 and end < len(text):
            end = start + boundary + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                _build_chunk(
                    document,
                    _with_heading_context(chunk_text, heading_path),
                    base_start + start,
                    base_start + end,
                    heading_path,
                    heading_level,
                    strategy,
                    fallback_used,
                    start_index + len(chunks),
                )
            )
        if end >= len(text):
            break
        start = max(0, end - settings.chunk_overlap)
    return chunks


def _with_heading_context(text: str, heading_path: list[str]) -> str:
    if not heading_path:
        return text
    return f"{' > '.join(heading_path)}\n\n{text}"


def _build_chunk(
    document: Document,
    text: str,
    start_char: int,
    end_char: int,
    heading_path: list[str],
    heading_level: int,
    strategy: str,
    fallback_used: bool,
    chunk_index: int,
) -> Chunk:
    chunk_hash = hashlib.sha256(f"{document.document_id}:{chunk_index}:{text}".encode("utf-8")).hexdigest()[:16]
    return Chunk(
        chunk_id=f"{document.document_id}-{chunk_hash}",
        document_id=document.document_id,
        title=document.title,
        text=text,
        start_char=start_char,
        end_char=end_char,
        metadata={
            **document.metadata,
            "chunk_index": chunk_index,
            "heading_path": heading_path,
            "heading_level": heading_level,
            "chunk_strategy": strategy,
            "fallback_used": fallback_used,
        },
    )


def _first_heading(markdown: str) -> str | None:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
    return None


def ingest_documents(path: Path, write: bool = True):
    """Public facade for the new multi-format document processing pipeline."""

    from ops_agent.services.document_processing.service import DocumentProcessingService

    return DocumentProcessingService().ingest(path, write=write)
