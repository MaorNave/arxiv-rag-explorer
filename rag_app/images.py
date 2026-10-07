"""Bonus feature: illustrate answers with an image from an external text-to-image API.

IMAGE_PROVIDER selects the backend:
    auto          (default) first configured of: openai, gemini, pollinations (with key),
                  huggingface; otherwise Pollinations' keyless tier (best effort)
    pollinations  POLLINATIONS_TOKEN -> FLUX via gen.pollinations.ai (free key at
                  https://enter.pollinations.ai); without a key the keyless legacy
                  endpoint is used, which is rate-limited per IP
    openai        OPENAI_API_KEY, gpt-image-2.5-flare (any OpenAI image model)
    gemini        GEMINI_API_KEY, gemini-nano-banana-2.1 ("Nano Banana")
    huggingface   HF_TOKEN, any HF text-to-image model (FLUX.1-schnell by default), routed by
                  huggingface_hub to whichever Inference Provider serves it
    none          feature disabled

Images are generated in a parallel LangGraph branch, so they never delay the text
answer. The bytes are stored under STORAGE_DIR/images and served at /generated/.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import random
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx

from .config import Settings

log = logging.getLogger(__name__)

PROVIDER_LABELS = {
    "pollinations": "Pollinations.ai",
    "openai": "OpenAI Images",
    "gemini": "Google Gemini",
    "huggingface": "Hugging Face Inference",
}
_DISABLED = {"", "none", "off", "disabled", "false"}
_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp", "image/gif": "gif"}
_KEEP_IMAGES = 200
_KEY_ADVICE = (
    "Get a free key at https://enter.pollinations.ai and set POLLINATIONS_TOKEN, or configure "
    "OPENAI_API_KEY / GEMINI_API_KEY / HF_TOKEN (see README)."
)
_KEYLESS_LIMITED = "Pollinations' keyless tier is rate-limited per IP (HTTP {status}). " + _KEY_ADVICE
_KEYLESS_DOWN = "Pollinations' keyless tier is best-effort and failed this time (HTTP {status}). " + _KEY_ADVICE


class ImageGenerationError(RuntimeError):
    pass


def resolve_provider(settings: Settings) -> str:
    provider = settings.image_provider
    if provider != "auto":
        return provider
    if settings.openai_api_key:
        return "openai"
    if settings.gemini_api_key:
        return "gemini"
    if settings.pollinations_token:
        return "pollinations"
    if settings.hf_token:
        return "huggingface"
    return "pollinations"


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
        self.provider = resolve_provider(settings)
        self.output_dir: Path = settings.images_dir
        self._transport = transport  # injectable for offline tests

    @property
    def keyless(self) -> bool:
        return self.provider == "pollinations" and not self.settings.pollinations_token

    @property
    def model(self) -> str | None:
        if self.keyless:
            return "keyless tier"
        return {
            "pollinations": self.settings.pollinations_model,
            "openai": self.settings.openai_image_model,
            "gemini": self.settings.gemini_image_model,
            "huggingface": self.settings.hf_image_model,
        }.get(self.provider)

    @property
    def unavailable_reason(self) -> str | None:
        s = self.settings
        if self.provider in _DISABLED:
            return "Image generation is disabled (IMAGE_PROVIDER=none)"
        if self.provider not in PROVIDER_LABELS:
            return f"Unknown IMAGE_PROVIDER '{self.provider}'"
        keys = {"openai": ("OPENAI_API_KEY", s.openai_api_key),
                "gemini": ("GEMINI_API_KEY", s.gemini_api_key),
                "huggingface": ("HF_TOKEN", s.hf_token)}
        if self.provider in keys and not keys[self.provider][1]:
            return f"Set {keys[self.provider][0]} to use the {PROVIDER_LABELS[self.provider]} provider"
        return None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    def describe(self) -> dict:
        return {
            "provider": self.provider,
            "label": PROVIDER_LABELS.get(self.provider, self.provider),
            "model": self.model,
            "keyless": self.keyless,
            "available": self.available,
            "reason": self.unavailable_reason,
        }

    async def generate(self, prompt: str) -> dict:
        if not self.available:
            raise ImageGenerationError(self.unavailable_reason)
        started = time.perf_counter()
        timeout = httpx.Timeout(self.settings.image_timeout_s, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, transport=self._transport) as client:
            handler = getattr(self, f"_{self.provider}")
            data, mime = await handler(client, prompt)
        if not data:
            raise ImageGenerationError("The provider returned an empty image")
        name = f"{uuid.uuid4().hex}.{_EXTENSIONS.get(mime, 'png')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / name).write_bytes(data)
        self._prune()
        return {
            "url": f"/generated/{name}",
            "prompt": prompt,
            "provider": self.provider,
            "model": self.model,
            "mime_type": mime,
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }

    # ------------------------------------------------------------------ providers
    @staticmethod
    def _check(response: httpx.Response) -> None:
        if response.status_code >= 400:
            detail = response.text[:300].strip()
            raise ImageGenerationError(f"HTTP {response.status_code} from image API: {detail}")

    @classmethod
    def _image_bytes(cls, response: httpx.Response) -> tuple[bytes, str]:
        cls._check(response)
        mime = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if not mime.startswith("image/"):
            raise ImageGenerationError(f"Expected an image but got '{mime or 'unknown'}': {response.text[:200]}")
        return response.content, mime

    async def _pollinations(self, client: httpx.AsyncClient, prompt: str) -> tuple[bytes, str]:
        seed = random.randint(1, 2**31 - 1)
        token = self.settings.pollinations_token
        if token:  # current API: FLUX & co., authenticated
            response = await client.get(
                f"https://gen.pollinations.ai/image/{quote(prompt, safe='')}",
                params={"model": self.settings.pollinations_model, "width": 1024, "height": 576, "seed": seed},
                headers={"Authorization": f"Bearer {token}"},
            )
        else:  # legacy keyless endpoint: works without sign-up but is rate-limited per IP
            response = await client.get(
                f"https://image.pollinations.ai/prompt/{quote(prompt, safe='')}",
                params={"width": 1024, "height": 576, "seed": seed, "nologo": "true", "private": "true"},
            )
            if response.status_code in (401, 402, 403, 429):
                raise ImageGenerationError(_KEYLESS_LIMITED.format(status=response.status_code))
            if response.status_code >= 500:
                raise ImageGenerationError(_KEYLESS_DOWN.format(status=response.status_code))
        return self._image_bytes(response)

    async def _openai(self, client: httpx.AsyncClient, prompt: str) -> tuple[bytes, str]:
        model = self.settings.openai_image_model
        payload: dict = {"model": model, "prompt": prompt, "n": 1}
        if model.startswith("dall-e"):
            payload.update(size="1792x1024" if model == "dall-e-3" else "1024x1024", response_format="b64_json")
        else:  # gpt-image-* always returns base64
            payload.update(size="1536x1024", quality=self.settings.openai_image_quality)
        response = await client.post(
            f"{self.settings.openai_base_url}/images/generations",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json=payload,
        )
        self._check(response)
        item = response.json()["data"][0]
        if item.get("b64_json"):
            return base64.b64decode(item["b64_json"]), "image/png"
        return self._image_bytes(await client.get(item["url"]))

    async def _gemini(self, client: httpx.AsyncClient, prompt: str) -> tuple[bytes, str]:
        url = (
            "https://generativelanguage.googleapis.com/v1/models/"
            f"{self.settings.gemini_image_model}:generateContent"
        )
        config: dict = {"responseModalities": ["TEXT", "IMAGE"]}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {**config, "responseFormat": {"image": {"aspectRatio": "16:9"}}},
        }
        headers = {"x-goog-api-key": self.settings.gemini_api_key}
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code == 400:  # older models may reject the aspect-ratio field: retry without it
            payload["generationConfig"] = config
            response = await client.post(url, headers=headers, json=payload)
        self._check(response)
        for candidate in response.json().get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                    return base64.b64decode(inline["data"]), mime
        raise ImageGenerationError("Gemini returned no image (the prompt may have been filtered)")

    async def _huggingface(self, client: httpx.AsyncClient, prompt: str) -> tuple[bytes, str]:
        """Hugging Face Inference Providers via the official client.

        HF_IMAGE_PROVIDER=auto lets huggingface_hub route the model to whichever provider
        serves it (FLUX.1-schnell currently runs on nscale), so no URL is hard-coded here.
        """
        from huggingface_hub import InferenceClient
        from huggingface_hub.errors import HfHubHTTPError

        def run() -> bytes:
            hf = InferenceClient(
                provider=self.settings.hf_image_provider,
                api_key=self.settings.hf_token,
                timeout=self.settings.image_timeout_s,
            )
            image = hf.text_to_image(prompt, model=self.settings.hf_image_model, width=1024, height=576)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()

        try:
            return await asyncio.to_thread(run), "image/png"
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 401:
                raise ImageGenerationError(
                    "Hugging Face rejected HF_TOKEN (HTTP 401): create a token with the "
                    "'Make calls to Inference Providers' permission"
                ) from exc
            if status == 402:
                raise ImageGenerationError(
                    "Hugging Face Inference Providers credits are exhausted (HTTP 402). Free accounts get no "
                    "monthly credits; buy credits or upgrade to PRO ($2/month included)"
                ) from exc
            raise ImageGenerationError(f"Hugging Face error (HTTP {status}): {exc}") from exc

    def _prune(self) -> None:
        try:
            files = sorted(self.output_dir.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in files[_KEEP_IMAGES:]:
                old.unlink(missing_ok=True)
        except OSError as exc:
            log.debug("Image pruning failed: %s", exc)
