#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_PATH="${REPO_ROOT}/apps/wfo_engine/app.py"

if [[ ! -f "${APP_PATH}" ]]; then
  echo "Error: WFO Engine app not found at ${APP_PATH}" >&2
  exit 1
fi

if command -v streamlit >/dev/null 2>&1; then
  exec streamlit run "${APP_PATH}" \
    --server.maxUploadSize=6144 \
    --server.maxMessageSize=6144 \
    "$@"
fi

if command -v python >/dev/null 2>&1; then
  exec python -m streamlit run "${APP_PATH}" \
    --server.maxUploadSize=6144 \
    --server.maxMessageSize=6144 \
    "$@"
fi

echo "Error: streamlit (or python) not found in PATH." >&2
echo "Install dependencies with: pip install -r ${REPO_ROOT}/apps/wfo_engine/requirements.txt" >&2
exit 1
