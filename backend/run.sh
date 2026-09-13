#!/usr/bin/env bash
# Start the recognition API. First run: python scripts/download_models.py
set -euo pipefail
cd "$(dirname "$0")"
exec uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "$@"
