from __future__ import annotations

import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ops_agent.services.document_processing import DocumentProcessingService


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    workdir = Path("./workfile")
    service = DocumentProcessingService()
    result = service.ingest(workdir)
    print(
        {
            "source_count": result.source_count,
            "chunk_count": result.chunk_count,
            "skipped_count": result.skipped_count,
            "failed": result.failed,
        }
    )


if __name__ == "__main__":
    main()
