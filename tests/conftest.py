<<<<<<< HEAD
import os
import sys


# Make project root importable so `import mmx` works when running `pytest`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
=======
from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Ensure project root and src/ are importable in tests."""

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    for p in (str(root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)
>>>>>>> eebb871 (version up)
