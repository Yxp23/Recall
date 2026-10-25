#!/bin/bash
# Run the Recall API server. Always from the repo root so models/ and data/ resolve.
set -e
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
export PYTHONPATH=backend
exec "$HOME/recall-env/bin/python" -m uvicorn recall.api.server:app --host 0.0.0.0 --port 8000
