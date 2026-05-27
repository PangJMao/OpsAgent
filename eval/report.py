from __future__ import annotations

from metrics import EvalResult


def print_console_report(results: list[EvalResult]) -> None:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    failed = total - passed
    pass_rate = (passed / total * 100) if total else 0.0

    print(f"总用例数：{total}")
    print(f"通过：{passed}")
    print(f"失败：{failed}")
    print(f"通过率：{pass_rate:.1f}%")

    failures = [result for result in results if not result.passed]
    if not failures:
        return

    print("\n失败详情：")
    for result in failures:
        print(f"[FAIL] {result.case_id}")
        print(f"问题：{result.question}")
        print("失败原因：")
        for reason in result.failures:
            print(f"- {reason}")
        if result.intent:
            print(f"识别意图：{result.intent}")
        if result.risk_level:
            print(f"风险等级：{result.risk_level}")
        preview = " ".join(result.answer.split())
        if preview:
            print(f"回答预览：{preview[:240]}")
        print()
