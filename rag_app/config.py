"""Runtime configuration.

Every setting can be overridden with an environment variable (or a ``.env`` file
in the project root). Defaults are tuned for a laptop with ~8 GB of RAM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Values already present in the environment always win over the .env file.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value is not None else default


def _env_optional_float(name: str) -> float | None:
    value = _env(name)
    return None if value is None or value.lower() == "auto" else float(value)


def _ollama_base_url() -> str:
    """OLLAMA_BASE_URL, else Ollama's own OLLAMA_HOST, normalised to a full URL."""
    raw = _env("OLLAMA_BASE_URL") or _env("OLLAMA_HOST") or "http://127.0.0.1:11434"
    if "://" not in raw:
        raw = f"http://{raw}"
    scheme, rest = raw.split("://", 1)
    host = rest.rstrip("/")
    if ":" not in host.split("/")[0]:
        host = f"{host}:11434"
    return f"{scheme}://{host}"


@dataclass(frozen=True)
class Settings:
    # Dataset & storage
    data_path: str = "data/arxiv_2.9k.jsonl"
    storage_dir: Path = PROJECT_ROOT / "storage"

    # Web server
    host: str = "127.0.0.1"
    port: int = 8080

    # Ollama models
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3.5:4b"
    embed_model: str = "nomic-embed-text"
    auto_pull_models: bool = True
    llm_temperature: float = 0.2
    llm_num_ctx: int = 8192
    llm_num_predict: int = 1024
    llm_thinking: bool = False
    llm_keep_alive: str = "30m"
    llm_timeout_s: float = 300.0

    # Retrieval
    top_k: int = 5
    fetch_k: int = 20
    # Fusion weights: None = auto (broad recall with the reranker, tuned precision without it)
    keyword_weight: float | None = None
    rewrite_weight: float | None = None
    query_rewrite: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_candidates: int = 20
    rerank_max_tokens: int = 384
    rerank_weight: float = 2.0
    analysis_timeout_s: float = 45.0
    chunk_size: int = 2000
    chunk_overlap: int = 200
    embed_batch_size: int = 64
    embed_doc_prefix: str | None = None
    embed_query_prefix: str | None = None

    # Hot-reload of the dataset file while the server runs
    watch_dataset: bool = True
    watch_interval_s: float = 10.0
    max_upload_mb: int = 512

    # Bonus: image generation, free options only (local Ollama, then free external APIs)
    image_provider: str = "auto"
    image_timeout_s: float = 120.0
    local_image_timeout_s: float = 600.0
    ollama_image_model: str = "x/flux2-klein:4b"
    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None
    cloudflare_image_model: str = "@cf/black-forest-labs/flux-1-schnell"
    pollinations_model: str = "flux"
    pollinations_token: str | None = None

    log_level: str = "INFO"

    @property
    def index_dir(self) -> Path:
        return self.storage_dir / "index"

    @property
    def images_dir(self) -> Path:
        return self.storage_dir / "images"

    @property
    def downloads_dir(self) -> Path:
        return self.storage_dir / "downloads"

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def models_dir(self) -> Path:
        return self.storage_dir / "models"

    @property
    def reranker_enabled(self) -> bool:
        return self.reranker_model.lower() not in ("", "none", "off", "false")

    @classmethod
    def from_env(cls) -> Settings:
        storage = Path(_env("STORAGE_DIR", str(PROJECT_ROOT / "storage"))).expanduser()
        if not storage.is_absolute():
            storage = PROJECT_ROOT / storage
        return cls(
            data_path=_env("DATA_PATH", cls.data_path),
            storage_dir=storage,
            host=_env("HOST", cls.host),
            port=_env_int("PORT", cls.port),
            ollama_base_url=_ollama_base_url(),
            llm_model=_env("LLM_MODEL", cls.llm_model),
            embed_model=_env("EMBED_MODEL", cls.embed_model),
            auto_pull_models=_env_bool("AUTO_PULL_MODELS", cls.auto_pull_models),
            llm_temperature=_env_float("LLM_TEMPERATURE", cls.llm_temperature),
            llm_num_ctx=_env_int("LLM_NUM_CTX", cls.llm_num_ctx),
            llm_num_predict=_env_int("LLM_NUM_PREDICT", cls.llm_num_predict),
            llm_thinking=_env_bool("LLM_THINKING", cls.llm_thinking),
            llm_keep_alive=_env("LLM_KEEP_ALIVE", cls.llm_keep_alive),
            llm_timeout_s=_env_float("LLM_TIMEOUT_S", cls.llm_timeout_s),
            top_k=_env_int("TOP_K", cls.top_k),
            fetch_k=_env_int("FETCH_K", cls.fetch_k),
            keyword_weight=_env_optional_float("KEYWORD_WEIGHT"),
            rewrite_weight=_env_optional_float("REWRITE_WEIGHT"),
            reranker_model=_env("RERANKER_MODEL", cls.reranker_model),
            rerank_candidates=_env_int("RERANK_CANDIDATES", cls.rerank_candidates),
            rerank_max_tokens=_env_int("RERANK_MAX_TOKENS", cls.rerank_max_tokens),
            rerank_weight=_env_float("RERANK_WEIGHT", cls.rerank_weight),
            query_rewrite=_env_bool("QUERY_REWRITE", cls.query_rewrite),
            analysis_timeout_s=_env_float("ANALYSIS_TIMEOUT_S", cls.analysis_timeout_s),
            chunk_size=_env_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", cls.chunk_overlap),
            embed_batch_size=_env_int("EMBED_BATCH_SIZE", cls.embed_batch_size),
            # Prefixes may legitimately be "" so they are read without the empty->default rule.
            embed_doc_prefix=os.environ.get("EMBED_DOC_PREFIX"),
            embed_query_prefix=os.environ.get("EMBED_QUERY_PREFIX"),
            watch_dataset=_env_bool("WATCH_DATASET", cls.watch_dataset),
            watch_interval_s=_env_float("WATCH_INTERVAL_S", cls.watch_interval_s),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", cls.max_upload_mb),
            image_provider=(_env("IMAGE_PROVIDER", cls.image_provider) or "none").lower(),
            image_timeout_s=_env_float("IMAGE_TIMEOUT_S", cls.image_timeout_s),
            local_image_timeout_s=_env_float("LOCAL_IMAGE_TIMEOUT_S", cls.local_image_timeout_s),
            ollama_image_model=_env("OLLAMA_IMAGE_MODEL", cls.ollama_image_model),
            cloudflare_account_id=_env("CLOUDFLARE_ACCOUNT_ID"),
            cloudflare_api_token=_env("CLOUDFLARE_API_TOKEN"),
            cloudflare_image_model=_env("CLOUDFLARE_IMAGE_MODEL", cls.cloudflare_image_model),
            pollinations_model=_env("POLLINATIONS_MODEL", cls.pollinations_model),
            pollinations_token=_env("POLLINATIONS_TOKEN") or _env("POLLINATIONS_API_KEY"),
            log_level=(_env("LOG_LEVEL", cls.log_level) or "INFO").upper(),
        )

    def ensure_dirs(self) -> None:
        for path in (self.index_dir, self.images_dir, self.downloads_dir, self.uploads_dir):
            path.mkdir(parents=True, exist_ok=True)
