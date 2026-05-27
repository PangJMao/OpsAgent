from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import yaml
except ImportError as exc:  # pragma: no cover - local environment has PyYAML
    raise SystemExit("缺少 PyYAML，请先安装 pyyaml 后再运行 eval。") from exc

from metrics import EvalResult, evaluate_case
from report import print_console_report


DEFAULT_CASES_PATH = Path(__file__).with_name("eval_cases.yaml")
DEFAULT_API_URL = "http://localhost:8000/rag/ask"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG / Rule RAG evaluation cases against local Agent API.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="Path to eval_cases.yaml")
    parser.add_argument("--api-url", default=None, help="Override API URL, e.g. http://localhost:8000/rag/ask")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds per case")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional sleep seconds between cases")
    args = parser.parse_args()

    config = _load_yaml(Path(args.cases))
    cases = config.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise SystemExit("eval cases 为空，请检查 eval_cases.yaml")

    api_url = args.api_url or os.getenv("EVAL_API_URL") or config.get("api_url") or DEFAULT_API_URL
    headers = _headers(config.get("headers") or {})

    results: list[EvalResult] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        response = _call_agent(api_url, str(case.get("question") or ""), headers, args.timeout)
        result = evaluate_case(case, response)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case_id}")
        if args.sleep:
            time.sleep(args.sleep)

    print()
    print_console_report(results)
    return 0 if all(result.passed for result in results) else 1


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"YAML 顶层必须是对象：{path}")
    return data


def _headers(config_headers: dict[str, Any]) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    for key, value in config_headers.items():
        headers[str(key)] = str(value)
    env_headers = os.getenv("EVAL_HEADERS")
    if env_headers:
        parsed = json.loads(env_headers)
        if not isinstance(parsed, dict):
            raise SystemExit("EVAL_HEADERS 必须是 JSON 对象")
        headers.update({str(key): str(value) for key, value in parsed.items()})
    return headers


def _call_agent(api_url: str, question: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    payload = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
    request = Request(api_url, data=payload, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {
            "answer": "",
            "refused": True,
            "error": f"HTTP {exc.code}: {detail}",
        }
    except URLError as exc:
        return {
            "answer": "",
            "refused": True,
            "error": f"API 不可用：{exc.reason}",
        }

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {"answer": body, "refused": False}

    if not isinstance(parsed, dict):
        return {"answer": str(parsed), "refused": False}
    return parsed


if __name__ == "__main__":
    sys.exit(main())
