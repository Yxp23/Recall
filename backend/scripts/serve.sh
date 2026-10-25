#!/bin/bash
# Run the Recall API server. Always from the repo root so models/ and data/ resolve.
set -e
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
export PYTHONPATH=backend
export RECALL_VOICE="${RECALL_VOICE:-1}"   # RECALL_VOICE=0 turns push-to-talk off (tap buttons still work)
export RECALL_DOOR="${RECALL_DOOR:-1}"   # RECALL_DOOR=0 turns the door camera off completely
# the virtualenv from the README; RECALL_PYTHON overrides it, and plain python3 is the last resort
PY="${RECALL_PYTHON:-$HOME/recall-env/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
exec "$PY" -m uvicorn recall.api.server:app --host 0.0.0.0 --port 8000
