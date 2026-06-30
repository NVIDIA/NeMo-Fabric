#!/bin/sh

python3 - <<'PY'
from pathlib import Path

path = Path('/app/calculator.py')
source = path.read_text()
updated = source.replace('return a - b', 'return a * b', 1)
if updated == source:
    raise SystemExit('expected multiply implementation was not found')
path.write_text(updated)
PY
