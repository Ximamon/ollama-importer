#!/usr/bin/env bash
set -euo pipefail

# Generic batch importer for generated LM Studio -> Ollama artifacts.
# Safety: dry mode by default. Set EXECUTE=1 to run real ollama create commands.
# Optional filter: NAME_FILTER='qwen' EXECUTE=1 ./import_all.sh ./ollama_imports

IMPORT_DIR="${1:-./ollama_imports}"
SELECTED_ONLY="${SELECTED_ONLY:-1}"
EXECUTE="${EXECUTE:-0}"
NAME_FILTER="${NAME_FILTER:-}"

if [[ "$SELECTED_ONLY" == "1" ]]; then
  INDEX_FILE="${INDEX_FILE:-$IMPORT_DIR/selected_models.json}"
else
  INDEX_FILE="${INDEX_FILE:-$IMPORT_DIR/index.json}"
fi

if [[ ! -f "$INDEX_FILE" ]]; then
  echo "Index file not found: $INDEX_FILE" >&2
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Warning: 'ollama' not found in PATH. Commands will fail if EXECUTE=1." >&2
fi

echo "Import dir   : $IMPORT_DIR"
echo "Index file   : $INDEX_FILE"
echo "Selected only: $SELECTED_ONLY"
echo "Execute      : $EXECUTE"
[[ -n "$NAME_FILTER" ]] && echo "Name filter  : $NAME_FILTER"

action_count=0

while IFS=$'\t' read -r name modelfile; do
  [[ -z "$name" || -z "$modelfile" ]] && continue

  if [[ -n "$NAME_FILTER" && "$name" != *"$NAME_FILTER"* ]]; then
    continue
  fi

  cmd=(ollama create "$name" -f "$modelfile")
  echo "> ${cmd[*]}"

  if [[ "$EXECUTE" == "1" ]]; then
    "${cmd[@]}"
  fi

  action_count=$((action_count + 1))
done < <(python3 - "$INDEX_FILE" <<'PY'
import json
import sys
from pathlib import Path

idx = Path(sys.argv[1])
data = json.loads(idx.read_text(encoding='utf-8'))

if isinstance(data, dict):
    items = data.get('models', [])
elif isinstance(data, list):
    items = data
else:
    items = []

for item in items:
    name = item.get('proposed_ollama_name')
    modelfile = item.get('modelfile_path')
    if not name or not modelfile:
        continue
    print(f"{name}\t{modelfile}")
PY
)

echo "Processed entries: $action_count"

if [[ "$EXECUTE" != "1" ]]; then
  echo "Dry mode active. Re-run with EXECUTE=1 to execute imports."
fi
