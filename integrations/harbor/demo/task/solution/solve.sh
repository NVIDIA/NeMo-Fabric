#!/bin/sh

python3 - <<'PY'
from pathlib import Path

path = Path('/app/calculator.py')
path.write_text(path.read_text().replace('return a - b', 'return a * b'))
PY
