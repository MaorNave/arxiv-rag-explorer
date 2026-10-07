"""Start the arXiv RAG Explorer web app on http://127.0.0.1:8080.

    python3 main.py                                  # uses data/arxiv_2.9k.jsonl
    DATA_PATH=/path/to/other.jsonl python3 main.py   # any .jsonl dataset
    python3 main.py --help                           # all CLI options

Nothing has to be installed by hand. On the first run this script creates a virtual
environment in .venv and installs requirements.txt into it (never into the system Python),
then the app sets up everything else inside storage/: its own Ollama if none is running,
Qwen 3.5, the embedding model, the reranker, and local FLUX images on Macs with >= 16 GB.
Later starts reuse all of it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIRED = "import fastapi, uvicorn, langgraph, langchain_ollama, langchain_chroma, chromadb, onnxruntime"


def _dependencies_installed(python: str = sys.executable) -> bool:
    return subprocess.call([python, "-c", REQUIRED], stderr=subprocess.DEVNULL) == 0


def _install_requirements(python: str) -> None:
    print("Installing the Python packages (first run only, a few minutes)…", flush=True)
    subprocess.check_call([python, "-m", "pip", "install", "--disable-pip-version-check", "-q", "--upgrade", "pip"])
    subprocess.check_call([python, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(ROOT / "requirements.txt")])


def bootstrap() -> None:
    """Make sure the dependencies are importable; set up .venv on the first run."""
    if sys.version_info < (3, 10):  # noqa: UP036 - friendly message for people running an old Python
        sys.exit(f"Python 3.10+ is required (this is {sys.version.split()[0]}).")
    if _dependencies_installed():
        return
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:  # an activated venv (e.g. PyCharm's) that lacks the packages: install into it
        _install_requirements(sys.executable)
        return
    if not VENV_PYTHON.exists():
        print(f"First run: creating a virtual environment in {VENV}", flush=True)
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    if not _dependencies_installed(str(VENV_PYTHON)):
        _install_requirements(str(VENV_PYTHON))
    # Continue inside the project's virtual environment.
    args = [str(VENV_PYTHON), str(ROOT / "main.py"), *sys.argv[1:]]
    if os.name == "nt":
        try:
            sys.exit(subprocess.call(args))
        except KeyboardInterrupt:
            sys.exit(130)
    os.execv(args[0], args)


if __name__ == "__main__":
    bootstrap()
    from rag_app.__main__ import main

    main()
