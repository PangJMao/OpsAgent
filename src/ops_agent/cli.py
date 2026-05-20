from __future__ import annotations

import argparse
import json
from pathlib import Path

from ops_agent.rag import RagPipeline, answer_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(prog="ops-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="导入 Markdown 或文本知识库文档。")
    ingest_parser.add_argument("path", type=Path)

    ask_parser = subparsers.add_parser("ask", help="基于本地 RAG 索引提问。")
    ask_parser.add_argument("question")

    args = parser.parse_args()
    pipeline = RagPipeline()

    if args.command == "ingest":
        result = pipeline.ingest(args.path)
    elif args.command == "ask":
        result = answer_to_dict(pipeline.ask(args.question))
    else:
        raise ValueError(f"未知命令：{args.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
