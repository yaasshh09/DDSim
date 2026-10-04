#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=${PYTHON:-python}; fi

CLIENT=(tests/unit/test_client.py tests/unit/test_browser_smoke.py tests/unit/test_vendor.py)

lint()   { "$PY" -m ruff check ddsim tests tools; }
fix()    { "$PY" -m ruff check --fix ddsim tests tools; }
format() { "$PY" -m ruff format ddsim tests tools; }
types()  { "$PY" -m mypy; }
test()   { "$PY" -m pytest "${CLIENT[@]/#/--ignore=}" "$@"; }
client() { "$PY" -m pytest "${CLIENT[@]}" "$@"; }
check()  { lint; types; test; client; }

case "${1:-}" in
  lint|fix|format|types|test|client|check) cmd=$1; shift; "$cmd" "$@" ;;
  *) echo "usage: ./run.sh {lint|fix|format|types|test|client|check} [pytest args]"; exit 2 ;;
esac
