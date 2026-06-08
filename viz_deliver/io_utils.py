"""JSON loading helpers."""
from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f'Missing data file: {path}')
    return json.loads(path.read_text(encoding='utf-8'))
