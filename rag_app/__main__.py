"""Command-line entry point: ``python -m rag_app`` (or ``python main.py``)."""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from . import __version__
from .config import Settings


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "posthog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="arXiv RAG Explorer: local RAG over research abstracts (LangGraph + Ollama/Qwen).",
    )
    parser.add_argument("--data-path", help="dataset .jsonl file, directory or URL (env: DATA_PATH)")
    parser.add_argument("--host", help="bind address (env: HOST, default 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port (env: PORT, default 8080)")
    parser.add_argument("--llm", help="Ollama chat model (env: LLM_MODEL, default qwen3.5:4b)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    for env, value in (("DATA_PATH", args.data_path), ("HOST", args.host), ("PORT", args.port), ("LLM_MODEL", args.llm)):
        if value is not None:
            os.environ[env] = str(value)

    settings = Settings.from_env()
    configure_logging(settings.log_level)
    log = logging.getLogger("rag_app")
    log.info("arXiv RAG Explorer %s", __version__)
    log.info("Dataset: %s | LLM: %s | embeddings: %s | images: %s",
             settings.data_path, settings.llm_model, settings.embed_model, settings.image_provider)
    log.info("Open http://%s:%d in your browser (API docs at /docs)", settings.host, settings.port)

    from .server import create_app  # imported late so logging is configured first

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
