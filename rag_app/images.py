"""Bonus feature: illustrate answers with a generated image, using free options only.

With IMAGE_PROVIDER=auto (default) the providers below are tried in this order, and if
one fails the next one is used:

    ollama        local and free: OLLAMA_IMAGE_MODEL (x/flux2-klein:4b) served by the same
                  Ollama that runs Qwen. Used when the model is installed and the Ollama
                  version can generate images (experimental image generation was temporarily
                  removed in Ollama 0.32.6; this switches on again by itself once it returns).
                  Runs *after* the answer so it never competes with the LLM for the GPU.
    cloudflare    free external API: Cloudflare Workers AI's free plan includes 10,000
                  neurons per day (about 170 FLUX.1 schnell images). Needs a free account:
                  CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN.
    pollinations  free external API: works without a key (rate-limited per IP, watermarked);
                  a free account key (POLLINATIONS_TOKEN) removes the watermark.

IMAGE_PROVIDER=ollama|cloudflare|pollinations forces a single provider; none disables the
feature. External providers run in a parallel LangGraph branch, so they never delay the
text answer either. Images are stored under STORAGE_DIR/images and served at /generated/.
"""

from __future__ import annotations

import base64
import logging
import random
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx

from .config import Settings

log = logging.getLogger(__name__)

PROVIDERS = ("ollama", "cloudflare", "pollinations")
PROVIDER_LABELS = {
    "ollama": "Ollama (local)",
    "cloudflare": "Cloudflare Workers AI",
    "pollinations": "Pollinations.ai",
}
_DISABLED = {"", "none", "off", "disabled", "false"}
_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
_KEEP_IMAGES = 200
_POLLINATIONS_ADVICE = (
    "A free account key (POLLINATIONS_TOKEN, https://enter.pollinations.ai) or a free Cloudflare "
    "Workers AI account (CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN) gives reliable images; see README."
)
_OLLAMA_UNSUPPORTED = (
    "This Ollama version cannot generate images (experimental image generation was removed in "
    "Ollama 0.32.6); using the free external APIs instead"
)


class ImageGenerationError(RuntimeError):
    pass


def _sniff_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def build_image_prompt(question: str, plan: dict | None = None) -> str:
    """Deterministic prompt (no LLM call, so the local model stays free for the answer)."""
    plan = plan or {}
    topic = (plan.get("topic") or "").strip() or question.strip().rstrip("?")
    keywords = [k for k in plan.get("keywords", []) if k.lower() not in topic.lower()][:4]
    prompt = f"Editorial illustration for a research article about {topic}"
    if keywords:
        prompt += f". Visual metaphor featuring {', '.join(keywords)}"
    prompt += (
        ". Isometric 3D style, clean composition, soft studio lighting, rich colors, highly detailed. "
        "No text, no letters, no logos, no watermark."
    )
    return prompt[:900]


class ImageGenerator:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.mode = settings.image_provider
        self.output_dir: Path = settings.images_dir
        self._transport = transport  # injectable for offline tests
        self.local_installed = False  # set by the service once Ollama has been checked
        self.local_unsupported = False  # set when this Ollama version refuses image generation

    # ------------------------------------------------------------------ provider selection
    @property
    def cloudflare_configured(self) -> bool:
        return bool(self.settings.cloudflare_account_id and self.settings.cloudflare_api_token)

    def chain(self) -> list[str]:
        """Providers to try, in order."""
        if self.mode in _DISABLED:
            return []
        if self.mode != "auto":
            return [self.mode] if self.mode in PROVIDERS else []
        chain = []
        if self.local_installed and not self.local_unsupported:
            chain.append("ollama")
        if self.cloudflare_configured:
            chain.append("cloudflare")
        chain.append("pollinations")
        return chain

    @property
    def provider(self) -> str | None:
        chain = self.chain()
        return chain[0] if chain else None

    @property
    def runs_after_answer(self) -> bool:
        """A local model shares the GPU with the LLM, so it waits until the answer is done."""
        return self.provider == "ollama"

    @property
    def unavailable_reason(self) -> str | None:
        if self.mode in _DISABLED:
            return "Image generation is disabled (IMAGE_PROVIDER=none)"
        if self.mode != "auto" and self.mode not in PROVIDERS:
            return f"Unknown IMAGE_PROVIDER '{self.mode}' (use auto, ollama, cloudflare, pollinations or none)"
        if self.mode == "cloudflare" and not self.cloudflare_configured:
            return "Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN to use Cloudflare Workers AI"
        if self.mode == "ollama" and self.local_unsupported:
            return _OLLAMA_UNSUPPORTED.split(";")[0]
        return None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    def model_for(self, provider: str | None) -> str | None:
        if provider == "pollinations" and not self.settings.pollinations_token:
            return "keyless tier"
        return {
            "ollama": self.settings.ollama_image_model,
            "cloudflare": self.settings.cloudflare_image_model.rsplit("/", 1)[-1],
            "pollinations": self.settings.pollinations_model,
        }.get(provider)

    def describe(self) -> dict:
        provider = self.provider
        return {
            "provider": provider,
            "label": PROVIDER_LABELS.get(provider, provider),
            "model": self.model_for(provider),
            "keyless": provider == "pollinations" and not self.settings.pollinations_token,
            "chain": [PROVIDER_LABELS[p] for p in self.chain()],
            "local": provider == "ollama",
            "runs_after_answer": self.runs_after_answer,
            "available": self.available,
            "reason": self.unavailable_reason,
        }

    # ------------------------------------------------------------------ generation
    async def generate(self, prompt: str) -> dict:
        if not self.available:
            raise ImageGenerationError(self.unavailable_reason)
        errors: list[str] = []
        for provider in self.chain():
            started = time.perf_counter()
            try:
                data, mime = await self._run(provider, prompt)
            except (ImageGenerationError, httpx.HTTPError, ValueError) as exc:
                message = str(exc) or type(exc).__name__
                log.info("Image provider %s failed: %s", provider, message)
                errors.append(f"{PROVIDER_LABELS[provider]}: {message}")
                continue
            return self._save(data, mime, prompt, provider, started)
        raise ImageGenerationError(" | ".join(errors) or "No image provider is available")

    async def _run(self, provider: str, prompt: str) -> tuple[bytes, str]:
        seconds = self.settings.local_image_timeout_s if provider == "ollama" else self.settings.image_timeout_s
        timeout = httpx.Timeout(seconds, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, transport=self._transport) as client:
            return await getattr(self, f"_{provider}")(client, prompt)

    def _save(self, data: bytes, mime: str, prompt: str, provider: str, started: float) -> dict:
        if not data:
            raise ImageGenerationError("The provider returned an empty image")
        name = f"{uuid.uuid4().hex}.{_EXTENSIONS.get(mime, 'png')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / name).write_bytes(data)
        self._prune()
        return {
            "url": f"/generated/{name}",
            "prompt": prompt,
            "provider": provider,
            "model": self.model_for(provider),
            "mime_type": mime,
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }

    # ------------------------------------------------------------------ providers
    @staticmethod
    def _check(response: httpx.Response) -> None:
        if response.status_code >= 400:
            raise ImageGenerationError(f"HTTP {response.status_code}: {response.text[:300].strip()}")

    async def _ollama(self, client: httpx.AsyncClient, prompt: str) -> tuple[bytes, str]:
        response = await client.post(
            f"{self.settings.ollama_base_url}/api/generate",
            json={
                "model": self.settings.ollama_image_model,
                "prompt": prompt,
                "width": 1024,
                "height": 576,
                "stream": False,
                "keep_alive": 0,  # unload right away so the LLM keeps its memory
            },
        )
        if response.status_code == 400 and "not currently supported" in response.text:
            self.local_unsupported = True  # skip Ollama until the app restarts (e.g. after an upgrade)
            raise ImageGenerationError(_OLLAMA_UNSUPPORTED)
        self._check(response)
        data = response.json().get("image")
        if not data:
            raise ImageGenerationError("Ollama returned no image")
        image = base64.b64decode(data)
        return image, _sniff_mime(image)

    async def _cloudflare(self, client: httpx.AsyncClient, prompt: str) -> tuple[bytes, str]:
        s = self.settings
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{s.cloudflare_account_id}"
            f"/ai/run/{s.cloudflare_image_model}"
        )
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {s.cloudflare_api_token}"},
            json={"prompt": prompt[:2048], "steps": 4},
        )
        if response.status_code in (401, 403):
            raise ImageGenerationError(
                f"Cloudflare rejected the API token (HTTP {response.status_code}); create one with the "
                "'Workers AI' permission"
            )
        if response.status_code == 429 or (response.status_code >= 400 and "neuron" in response.text.lower()):
            raise ImageGenerationError(
                "Cloudflare's free daily allowance (10,000 neurons) is used up; it resets every day"
            )
        self._check(response)
        if response.headers.get("content-type", "").startswith("image/"):
            return response.content, response.headers["content-type"].split(";")[0]
        body = response.json()
        result = body.get("result") if isinstance(body.get("result"), dict) else body
        data = result.get("image")
        if not data:
            raise ImageGenerationError(f"Cloudflare returned no image: {str(body)[:200]}")
        image = base64.b64decode(data)
        return image, _sniff_mime(image)

    async def _pollinations(self, client: httpx.AsyncClient, prompt: str) -> tuple[bytes, str]:
        seed = random.randint(1, 2**31 - 1)
        token = self.settings.pollinations_token
        if token:  # free account key: authenticated API, no watermark
            response = await client.get(
                f"https://gen.pollinations.ai/image/{quote(prompt, safe='')}",
                params={"model": self.settings.pollinations_model, "width": 1024, "height": 576, "seed": seed},
                headers={"Authorization": f"Bearer {token}"},
            )
        else:  # keyless endpoint: no sign-up, but rate-limited per IP
            response = await client.get(
                f"https://image.pollinations.ai/prompt/{quote(prompt, safe='')}",
                params={"width": 1024, "height": 576, "seed": seed, "nologo": "true", "private": "true"},
            )
            if response.status_code in (401, 402, 403, 429):
                raise ImageGenerationError(
                    f"keyless tier is rate-limited per IP (HTTP {response.status_code}). {_POLLINATIONS_ADVICE}"
                )
            if response.status_code >= 500:
                raise ImageGenerationError(
                    f"keyless tier is best-effort and failed (HTTP {response.status_code}). {_POLLINATIONS_ADVICE}"
                )
        self._check(response)
        mime = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if not mime.startswith("image/"):
            raise ImageGenerationError(f"Expected an image but got '{mime or 'unknown'}'")
        return response.content, mime

    def _prune(self) -> None:
        try:
            files = sorted(self.output_dir.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in files[_KEEP_IMAGES:]:
                old.unlink(missing_ok=True)
        except OSError as exc:
            log.debug("Image pruning failed: %s", exc)
