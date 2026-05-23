#!/usr/bin/env python3
"""Fix mmcv/parallel/_functions.py for PyTorch >= 2.1 (SegMAN README).

Does NOT import mmcv (broken _functions.py would block import).
"""
from __future__ import annotations

import sys
from pathlib import Path


def find_functions_py() -> Path:
    candidates = []
    try:
        import site
        for base in site.getsitepackages():
            candidates.append(Path(base) / "mmcv" / "parallel" / "_functions.py")
    except Exception:
        pass
    for p in sys.path:
        candidates.append(Path(p) / "mmcv" / "parallel" / "_functions.py")
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "mmcv/parallel/_functions.py not found; activate segman and check install"
    )


PATCH_BODY = """            from packaging import version
            if version.parse(torch.__version__) >= version.parse('2.1.0'):
                streams = [
                    _get_stream(torch.device('cuda', device))
                    for device in target_gpus
                ]
            else:
                streams = [_get_stream(device) for device in target_gpus]
"""

MARKER_START = "# Perform CPU to GPU copies in a background stream"
MARKER_END = "outputs = scatter(input, target_gpus, streams)"


def main() -> int:
    target = find_functions_py()
    text = target.read_text(encoding="utf-8")
    start = text.find(MARKER_START)
    end = text.find(MARKER_END)
    if start == -1 or end == -1 or end <= start:
        print("Could not locate patch region in _functions.py", file=sys.stderr)
        print(f"File: {target}", file=sys.stderr)
        return 1

    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.rfind("\n", 0, end) + 1
    new_text = text[:line_start] + MARKER_START + "\n" + PATCH_BODY + text[line_end:]

    if new_text == text:
        print("No changes written (already patched?).")
    else:
        target.write_text(new_text, encoding="utf-8")
        print(f"Patched: {target}")

    try:
        compile(new_text, str(target), "exec")
        print("Syntax check: OK")
    except SyntaxError as e:
        print(f"Syntax check FAILED: {e}", file=sys.stderr)
        return 1

    print('Run: python -c "from mmcv.parallel import collate; print(\'mmcv ok\')"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
