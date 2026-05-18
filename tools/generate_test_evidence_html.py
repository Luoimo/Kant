from __future__ import annotations

from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
JUNIT_XML = ROOT / "backend" / "core-results.xml"
COVERAGE_XML = ROOT / "backend" / "coverage.xml"
OUTPUT = ROOT / "docs" / "test_evidence_report.html"


def parse_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = []

    if root.tag == "testsuite":
        suite_nodes = [root]
    else:
        suite_nodes = list(root.findall(".//testsuite"))

    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    for suite in suite_nodes:
        item = {
            "name": suite.attrib.get("name", "test suite"),
            "tests": int(float(suite.attrib.get("tests", 0))),
            "failures": int(float(suite.attrib.get("failures", 0))),
            "errors": int(float(suite.attrib.get("errors", 0))),
            "skipped": int(float(suite.attrib.get("skipped", 0))),
            "time": float(suite.attrib.get("time", 0.0)),
        }
        for key in totals:
            totals[key] += item[key]
        suites.append(item)

    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    return {"totals": totals, "suites": suites}


def parse_coverage(path: Path) -> dict:
    root = ET.parse(path).getroot()
    line_rate = float(root.attrib.get("line-rate", 0.0)) * 100
    branch_rate = float(root.attrib.get("branch-rate", 0.0)) * 100
    lines_valid = int(float(root.attrib.get("lines-valid", 0)))
    lines_covered = int(float(root.attrib.get("lines-covered", 0)))
    branches_valid = int(float(root.attrib.get("branches-valid", 0)))
    branches_covered = int(float(root.attrib.get("branches-covered", 0)))

    packages = []
    for pkg in root.findall(".//package"):
        packages.append({
            "name": pkg.attrib.get("name", "package"),
            "line_rate": float(pkg.attrib.get("line-rate", 0.0)) * 100,
            "branch_rate": float(pkg.attrib.get("branch-rate", 0.0)) * 100,
        })
    packages.sort(key=lambda x: x["line_rate"], reverse=True)

    return {
        "line_rate": line_rate,
        "branch_rate": branch_rate,
        "lines_valid": lines_valid,
        "lines_covered": lines_covered,
        "branches_valid": branches_valid,
        "branches_covered": branches_covered,
        "packages": packages,
    }


def status_badge(totals: dict) -> tuple[str, str]:
    if totals["failures"] or totals["errors"]:
        return "Failed", "bad"
    if totals["skipped"]:
        return "Passed with skipped tests", "warn"
    return "Passed", "good"


def build_html() -> str:
    junit = parse_junit(JUNIT_XML)
    coverage = parse_coverage(COVERAGE_XML)
    totals = junit["totals"]
    label, class_name = status_badge(totals)

    suite_rows = "\n".join(
        f"<tr><td>{escape(s['name'])}</td><td>{s['tests']}</td><td>{s['tests'] - s['failures'] - s['errors'] - s['skipped']}</td>"
        f"<td>{s['failures']}</td><td>{s['errors']}</td><td>{s['skipped']}</td><td>{s['time']:.2f}s</td></tr>"
        for s in junit["suites"]
    )
    package_rows = "\n".join(
        f"<tr><td>{escape(p['name'])}</td><td>{p['line_rate']:.1f}%</td><td>{p['branch_rate']:.1f}%</td></tr>"
        for p in coverage["packages"][:16]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kant Test Evidence Report</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #64748b;
      --line: #dbe5ef;
      --soft: #f4f7fb;
      --blue: #1f4e79;
      --good: #0f766e;
      --warn: #b45309;
      --bad: #b91c1c;
    }}
    body {{
      margin: 0;
      font: 15px/1.5 Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: #fff;
    }}
    main {{
      max-width: 1080px;
      margin: 40px auto 64px;
      padding: 0 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      color: var(--blue);
      font-size: 32px;
    }}
    .subtitle {{
      margin: 0 0 28px;
      color: var(--muted);
      font-size: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 18px 0 30px;
    }}
    .card {{
      border: 1px solid var(--line);
      background: var(--soft);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric {{
      display: block;
      font-size: 28px;
      font-weight: 700;
      color: var(--blue);
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
    }}
    .badge {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      color: #fff;
      font-weight: 700;
      font-size: 13px;
    }}
    .good {{ background: var(--good); }}
    .warn {{ background: var(--warn); }}
    .bad {{ background: var(--bad); }}
    h2 {{
      margin-top: 30px;
      color: var(--blue);
      border-bottom: 2px solid var(--line);
      padding-bottom: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 24px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #edf3f9;
      color: #153d61;
    }}
    .note {{
      border-left: 4px solid var(--blue);
      background: #f8fafc;
      padding: 12px 14px;
      color: #334155;
    }}
    code {{
      font-family: "Courier New", monospace;
      color: #475569;
    }}
  </style>
</head>
<body>
<main>
  <h1>Kant Test Evidence Report</h1>
  <p class="subtitle">Readable summary generated from <code>backend/core-results.xml</code> and <code>backend/coverage.xml</code>.</p>

  <p><span class="badge {class_name}">{label}</span></p>

  <section class="grid">
    <div class="card"><span class="metric">{totals['tests']}</span><span class="label">Total Tests</span></div>
    <div class="card"><span class="metric">{totals['passed']}</span><span class="label">Passed</span></div>
    <div class="card"><span class="metric">{coverage['line_rate']:.1f}%</span><span class="label">Line Coverage</span></div>
    <div class="card"><span class="metric">{coverage['branch_rate']:.1f}%</span><span class="label">Branch Coverage</span></div>
  </section>

  <h2>Automated Test Summary</h2>
  <table>
    <thead><tr><th>Test Suite</th><th>Total</th><th>Passed</th><th>Failures</th><th>Errors</th><th>Skipped</th><th>Time</th></tr></thead>
    <tbody>{suite_rows}</tbody>
  </table>

  <h2>Coverage Summary</h2>
  <table>
    <tbody>
      <tr><th>Lines Covered</th><td>{coverage['lines_covered']} / {coverage['lines_valid']}</td></tr>
      <tr><th>Branches Covered</th><td>{coverage['branches_covered']} / {coverage['branches_valid']}</td></tr>
      <tr><th>Overall Line Coverage</th><td>{coverage['line_rate']:.1f}%</td></tr>
      <tr><th>Overall Branch Coverage</th><td>{coverage['branch_rate']:.1f}%</td></tr>
    </tbody>
  </table>

  <h2>Package Coverage Snapshot</h2>
  <table>
    <thead><tr><th>Package</th><th>Line Coverage</th><th>Branch Coverage</th></tr></thead>
    <tbody>{package_rows}</tbody>
  </table>

  <h2>Assessment Relevance</h2>
  <table>
    <thead><tr><th>Evidence</th><th>What It Supports</th><th>Assessment Area</th></tr></thead>
    <tbody>
      <tr><td>Automated backend test results</td><td>Validates core backend, RAG, agent, API, and security behaviours.</td><td>Implementation and Testing</td></tr>
      <tr><td>Coverage report</td><td>Shows that backend code was exercised by automated tests.</td><td>LLMSecOps Quality Control</td></tr>
      <tr><td>Security test cases</td><td>Demonstrates prompt injection and unsafe input handling.</td><td>AI Security</td></tr>
      <tr><td>Citation and retrieval tests</td><td>Supports evidence-grounded answer generation.</td><td>Explainable and Responsible AI</td></tr>
    </tbody>
  </table>

  <p class="note">This HTML file is intended as a human-readable evidence summary. The raw XML files remain useful as machine-readable CI/CD artifacts.</p>
</main>
</body>
</html>
"""


def main() -> None:
    OUTPUT.write_text(build_html(), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
