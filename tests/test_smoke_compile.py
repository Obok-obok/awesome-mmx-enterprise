import os
import py_compile


def test_all_python_files_compile():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for dirpath, _, filenames in os.walk(root):
        # Skip venv or cache-like folders
        if any(p in dirpath for p in [".venv", "__pycache__", ".git", "out/runs"]):
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                py_compile.compile(os.path.join(dirpath, fn), doraise=True)
