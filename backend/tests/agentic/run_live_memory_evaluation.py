from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
OUTPUT = ROOT / "artifacts" / "agentic-ai" / "memory-live-results.json"
sys.path.insert(0, str(BACKEND))


def main() -> None:
    from memory.mem0_store import Mem0Store

    suffix = uuid.uuid4().hex[:10]
    user_a = f"agent-eval-a-{suffix}"
    user_b = f"agent-eval-b-{suffix}"
    marker = f"KANT_EVAL_CITATION_PREF_{suffix}"
    store = Mem0Store()
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "enabled": bool(getattr(store, "_enabled", False)),
        "same_user_recall": False,
        "cross_user_isolation": False,
        "deletion_effective": False,
    }

    try:
        store.add_qa(
            user_id=user_a,
            query=f"I prefer source-based answers. Remember {marker}.",
            answer="Preference acknowledged.",
        )
        time.sleep(3)
        same_user = store.search(
            user_id=user_a,
            query="What answer style do I prefer?",
            top_k=5,
        )
        other_user = store.search(
            user_id=user_b,
            query="What answer style do I prefer?",
            top_k=5,
        )
        result.update(
            {
                "same_user_result_count": len(same_user),
                "other_user_result_count": len(other_user),
                "same_user_recall": any(marker in item for item in same_user),
                "cross_user_isolation": not any(marker in item for item in other_user),
            }
        )
    finally:
        store.delete_all(user_id=user_a)
        store.delete_all(user_id=user_b)
        time.sleep(1)
        after_delete = store.search(user_id=user_a, query=marker, top_k=5)
        result["after_delete_result_count"] = len(after_delete)
        result["deletion_effective"] = not any(marker in item for item in after_delete)

    result["passed"] = all(
        result[key]
        for key in ["enabled", "same_user_recall", "cross_user_isolation", "deletion_effective"]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
