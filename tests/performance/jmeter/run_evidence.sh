#!/usr/bin/env bash
set -euo pipefail

if ! command -v jmeter >/dev/null 2>&1; then
  echo "Apache JMeter is required. Install JMeter 5.6.3 or run the DevSecOps GitHub Actions workflow." >&2
  exit 1
fi

protocol="${1:-http}"
host="${2:-127.0.0.1}"
port="${3:-8000}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
root_dir="$(cd "$(dirname "$0")/../../.." && pwd)"
test_plan="$root_dir/tests/performance/jmeter/kant-api-performance.jmx"
output_root="$root_dir/docs/final_evidence_package/04-devsecops/jmeter/$run_id"
python_bin="${PYTHON_BIN:-$root_dir/.venv312/bin/python}"
seed_email="${PERF_EMAIL:-kant.jmeter.reader@example.com}"
test_password="${PERF_PASSWORD:-Kant-JMeter-2026!}"
admin_email="${PERF_ADMIN_EMAIL:-kant.jmeter.admin@example.com}"
admin_password="${PERF_ADMIN_PASSWORD:-$test_password}"
book_id="${PERF_BOOK_ID:-kant-performance-book}"

mkdir -p "$output_root/load" "$output_root/stress"

"$python_bin" "$root_dir/tests/performance/seed_performance_data.py" \
  --email "$seed_email" --password "$test_password" \
  --admin-email "$admin_email" --admin-password "$admin_password" \
  --book-id "$book_id" \
  > "$output_root/seed.json"
seed_user_id="$("$python_bin" -c 'import json, sys; print(json.load(open(sys.argv[1]))["user_id"])' "$output_root/seed.json")"

jmeter -n -t "$test_plan" \
  -Jprotocol="$protocol" -Jhost="$host" -Jport="$port" \
  -Jrun_id="${run_id}-load" \
  -Jseed_email="$seed_email" -Jtest_password="$test_password" \
  -Jadmin_email="$admin_email" -Jadmin_password="$admin_password" \
  -Jseed_user_id="$seed_user_id" -Jbook_id="$book_id" \
  -Jauth_threads=20 -Jauth_ramp_up=10 \
  -Jreader_threads=20 -Jreader_ramp_up=20 -Jduration=120 \
  -l "$output_root/load/results.jtl" \
  -e -o "$output_root/load/dashboard"

jmeter -n -t "$test_plan" \
  -Jprotocol="$protocol" -Jhost="$host" -Jport="$port" \
  -Jrun_id="${run_id}-stress" \
  -Jseed_email="$seed_email" -Jtest_password="$test_password" \
  -Jadmin_email="$admin_email" -Jadmin_password="$admin_password" \
  -Jseed_user_id="$seed_user_id" -Jbook_id="$book_id" \
  -Jauth_threads=100 -Jauth_ramp_up=20 \
  -Jreader_threads=100 -Jreader_ramp_up=20 -Jduration=90 \
  -l "$output_root/stress/results.jtl" \
  -e -o "$output_root/stress/dashboard"

echo "$output_root"
