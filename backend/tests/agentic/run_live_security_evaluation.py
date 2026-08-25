from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
CASES = Path(__file__).with_name("cases.json")
OUTPUT = ROOT / "artifacts" / "agentic-ai" / "security-live-results.json"
sys.path.insert(0, str(BACKEND))


def main() -> None:
    from security.input_filter import run_lakera_guard_check

    cases = [
        case
        for case in json.loads(CASES.read_text(encoding="utf-8"))
        if case["category"] == "security"
    ]
    results = []
    for case in cases:
        started = time.perf_counter()
        result = run_lakera_guard_check(case["input"])
        expected = case["expected"]
        allowed_pass = result.allowed is bool(expected["allowed"])
        risk_pass = (
            True
            if expected["allowed"]
            else expected.get("risk") in result.categories or "lakera_flagged" in result.categories
        )
        results.append(
            {
                "case_id": case["id"],
                "input": case["input"],
                "expected_allowed": expected["allowed"],
                "actual_allowed": result.allowed,
                "expected_risk": expected.get("risk"),
                "actual_categories": result.categories,
                "reason": result.reason,
                "allowed_pass": allowed_pass,
                "risk_pass": risk_pass,
                "passed": allowed_pass and risk_pass,
                "latency_seconds": round(time.perf_counter() - started, 3),
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": len(results),
        "passed": sum(row["passed"] for row in results),
        "failed": sum(not row["passed"] for row in results),
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
