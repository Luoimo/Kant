from __future__ import annotations

import sys

from backend.agents.orchestrator_agent import run_minimal_graph


def main() -> None:
    # Windows 终端默认编码可能导致中文乱码，这里尽量统一为 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    query = "克尔凯郭尔对焦虑的定义是什么？"
    state = run_minimal_graph(query)

    print("\n=== Answer ===\n")
    print(state.get("answer", ""))

    print("\n=== Citations ===\n")
    citations = state.get("citations") or []
    for i, c in enumerate(citations, start=1):
        sections = getattr(c, "section_indices", None) or []
        print(f"[{i}] source={getattr(c, 'source', '')} sections={sections}")
        snippet = getattr(c, "snippet", None)
        if snippet:
            print(f"    {snippet}\n")


if __name__ == "__main__":
    main()

