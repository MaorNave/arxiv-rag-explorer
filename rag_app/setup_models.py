"""Ollama helpers: reachability check, model download and capability detection.

Also a small, optional CLI that downloads the essentials up front (the app does it on its
first start anyway): the Qwen chat model and the embedding model through Ollama, plus
the small cross-encoder reranker from Hugging Face. If no Ollama is running, the app's
own copy is installed into STORAGE_DIR/runtime and used, like `python main.py` does.
The optional local image model is set up by the app itself, in the background.

    python -m rag_app.setup_models                    # LLM_MODEL + EMBED_MODEL from env/.env
    python -m rag_app.setup_models --llm qwen3.5:9b   # pick another Qwen size
"""

from __future__ import annotations

import argparse
import signal
import sys
from collections.abc import Callable, Iterable

import httpx
import ollama

from .config import Settings
from .runtime import RuntimeManager

ProgressCallback = Callable[[str, "float | None", str], None]

INSTALL_HINT = (
    "Ollama is not reachable at {url}. With a local OLLAMA_BASE_URL the app installs and runs its "
    "own copy automatically (MANAGED_OLLAMA=true); for a remote server, make sure it is running there."
)


def normalize(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


def ollama_version(base_url: str, timeout: float = 3.0) -> str | None:
    try:
        response = httpx.get(f"{base_url}/api/version", timeout=timeout)
        response.raise_for_status()
        return response.json().get("version", "unknown")
    except (httpx.HTTPError, ValueError):
        return None


def local_models(base_url: str) -> set[str]:
    client = ollama.Client(host=base_url)
    return {normalize(m.model) for m in client.list().models if m.model}


def model_capabilities(base_url: str, model: str) -> list[str]:
    try:
        return list(ollama.Client(host=base_url).show(model).capabilities or [])
    except Exception:
        return []


def pull_model(base_url: str, model: str, on_progress: ProgressCallback | None = None) -> None:
    client = ollama.Client(host=base_url)
    for part in client.pull(model, stream=True):
        if on_progress is None:
            continue
        percent = None
        if part.total and part.completed is not None:
            percent = round(100 * part.completed / part.total, 1)
        on_progress(model, percent, part.status or "")


def ensure_models(
    base_url: str, models: Iterable[str], *, pull: bool = True, on_progress: ProgressCallback | None = None
) -> list[str]:
    """Pull every missing model; returns the list of models that were downloaded."""
    have = local_models(base_url)
    missing = [m for m in dict.fromkeys(models) if normalize(m) not in have]
    if missing and not pull:
        commands = "; ".join(f"ollama pull {m}" for m in missing)
        raise RuntimeError(f"Missing Ollama models: {', '.join(missing)}. Run: {commands}")
    for model in missing:
        pull_model(base_url, model, on_progress)
    return missing


def cli_progress() -> ProgressCallback:
    last: dict[str, str] = {}

    def report(model: str, percent: float | None, status: str) -> None:
        line = f"  {model}: {status}" + (f" {percent:5.1f}%" if percent is not None else "")
        if last.get(model) != line:
            last[model] = line
            sys.stdout.write("\r" + line.ljust(78))
            sys.stdout.flush()

    return report


def connect(settings: Settings, runtime: RuntimeManager, models: list[str], base_url: str | None = None) -> str | None:
    """Ollama version at base_url for a command-line tool, running the app's own copy there when
    nothing is (like `python main.py`; stop it with runtime.stop()). None, with a hint, if unreachable."""
    base_url = base_url or settings.ollama_base_url
    version = ollama_version(base_url)
    if version is None and base_url == settings.ollama_base_url and runtime.can_run_main():
        print("No Ollama running: setting up the app's own copy in", runtime.dir)
        progress = cli_progress()
        runtime.start_main(models, lambda pct, text: progress("ollama", pct, text))
        print()
        version = ollama_version(base_url)
    if version is None:
        print(INSTALL_HINT.format(url=base_url), file=sys.stderr)
    return version


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Download the Ollama models used by the RAG app.")
    parser.add_argument("--llm", default=settings.llm_model, help=f"chat model (default: {settings.llm_model})")
    parser.add_argument("--embed", default=settings.embed_model, help=f"embedding model (default: {settings.embed_model})")
    parser.add_argument("--base-url", default=settings.ollama_base_url, help="Ollama server URL")
    parser.add_argument("--no-reranker", action="store_true", help="skip the cross-encoder download")
    args = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # `kill` unwinds too, through the finally below
    runtime = RuntimeManager(settings)
    try:
        return _setup(args, settings, runtime)
    finally:
        runtime.stop()  # an Ollama started by this command never outlives it (errors, Ctrl+C, kill)


def _setup(args: argparse.Namespace, settings: Settings, runtime: RuntimeManager) -> int:
    models = [args.llm, args.embed]
    version = connect(settings, runtime, models, args.base_url)
    if version is None:
        return 1
    print(f"Ollama {version} at {args.base_url}")
    pulled = ensure_models(args.base_url, models, on_progress=cli_progress())
    if pulled:
        print()
    for model in models:
        state = "downloaded" if model in pulled else "already present"
        print(f"  ✓ {model} ({state})")
    if settings.reranker_enabled and not args.no_reranker:
        from .reranker import CrossEncoderReranker

        print(f"  … reranker {settings.reranker_model} (Hugging Face, ~90 MB on first run)")
        CrossEncoderReranker(settings.reranker_model, settings.models_dir, settings.rerank_max_tokens).load()
        print(f"  ✓ {settings.reranker_model}")
    print("Models are ready. Start the app with:  python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
