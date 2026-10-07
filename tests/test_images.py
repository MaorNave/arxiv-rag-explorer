"""Image providers, tested offline with httpx.MockTransport (no keys, no network)."""

import asyncio
import base64
import json
from dataclasses import replace

import httpx
import pytest

from rag_app.images import ImageGenerationError, ImageGenerator, resolve_provider

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def generator(settings, handler, **overrides):
    return ImageGenerator(replace(settings, **overrides), transport=httpx.MockTransport(handler))


def test_auto_provider_resolution(settings):
    auto = replace(settings, image_provider="auto")
    assert resolve_provider(auto) == "pollinations"  # keyless fallback
    assert resolve_provider(replace(auto, hf_token="hf")) == "huggingface"
    assert resolve_provider(replace(auto, hf_token="hf", gemini_api_key="g")) == "gemini"
    assert resolve_provider(replace(auto, gemini_api_key="g", openai_api_key="o")) == "openai"
    assert not ImageGenerator(replace(settings, image_provider="none")).available
    assert "OPENAI_API_KEY" in ImageGenerator(replace(settings, image_provider="openai")).unavailable_reason


def test_keyless_pollinations_success_saves_the_image(settings):
    def handler(request):
        assert request.url.host == "image.pollinations.ai" and "Authorization" not in request.headers
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

    gen = generator(settings, handler, image_provider="pollinations")
    image = asyncio.run(gen.generate("a test prompt"))
    assert image["url"].startswith("/generated/") and image["url"].endswith(".png")
    assert (settings.images_dir / image["url"].split("/")[-1]).read_bytes() == PNG
    assert gen.describe()["keyless"] is True


@pytest.mark.parametrize("status, phrase", [(402, "rate-limited"), (429, "rate-limited"), (500, "best-effort")])
def test_keyless_pollinations_errors_are_actionable(settings, status, phrase):
    gen = generator(settings, lambda r: httpx.Response(status, json={"error": "x"}), image_provider="pollinations")
    with pytest.raises(ImageGenerationError, match=phrase):
        asyncio.run(gen.generate("prompt"))


def test_pollinations_with_token_uses_the_authenticated_api(settings):
    def handler(request):
        assert request.url.host == "gen.pollinations.ai"
        assert request.headers["Authorization"] == "Bearer sk_test"
        assert request.url.params["model"] == "flux"
        return httpx.Response(200, content=PNG, headers={"content-type": "image/jpeg"})

    gen = generator(settings, handler, image_provider="auto", pollinations_token="sk_test")
    assert asyncio.run(gen.generate("prompt"))["url"].endswith(".jpg")


def test_openai_provider(settings):
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/images/generations" and body["model"] == "gpt-image-2.5-flare"
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})

    gen = generator(settings, handler, image_provider="openai", openai_api_key="sk")
    assert asyncio.run(gen.generate("prompt"))["provider"] == "openai"


GEMINI_IMAGE = {"candidates": [{"content": {"parts": [
    {"text": "here"}, {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(PNG).decode()}},
]}}]}


def test_gemini_provider(settings):
    def handler(request):
        assert request.headers["x-goog-api-key"] == "g"
        assert request.url.path == "/v1/models/gemini-nano-banana-2.1:generateContent"
        config = json.loads(request.content)["generationConfig"]
        assert config["responseFormat"] == {"image": {"aspectRatio": "16:9"}}
        return httpx.Response(200, json=GEMINI_IMAGE)

    gen = generator(settings, handler, image_provider="gemini", gemini_api_key="g")
    assert asyncio.run(gen.generate("prompt"))["mime_type"] == "image/png"


def test_gemini_retries_without_aspect_ratio_when_rejected(settings):
    calls = []

    def handler(request):
        config = json.loads(request.content)["generationConfig"]
        calls.append(config)
        if "responseFormat" in config:
            return httpx.Response(400, json={"error": {"message": "Unknown name responseFormat"}})
        return httpx.Response(200, json=GEMINI_IMAGE)

    gen = generator(settings, handler, image_provider="gemini", gemini_api_key="g",
                    gemini_image_model="gemini-3-pro-image")
    asyncio.run(gen.generate("prompt"))
    assert len(calls) == 2 and calls[1] == {"responseModalities": ["TEXT", "IMAGE"]}


class FakeInferenceClient:
    """Stands in for huggingface_hub.InferenceClient (no network)."""

    status = None
    seen: dict = {}

    def __init__(self, provider=None, api_key=None, timeout=None):
        FakeInferenceClient.seen = {"provider": provider, "api_key": api_key}

    def text_to_image(self, prompt, *, model, width, height):
        from huggingface_hub.errors import HfHubHTTPError

        FakeInferenceClient.seen.update(model=model, size=(width, height))
        if self.status:
            response = httpx.Response(self.status, request=httpx.Request("POST", "https://router.huggingface.co"))
            raise HfHubHTTPError("error", response=response)
        from PIL import Image

        return Image.new("RGB", (8, 8), "purple")


def test_huggingface_provider_uses_the_routed_client(settings, monkeypatch):
    monkeypatch.setattr("huggingface_hub.InferenceClient", FakeInferenceClient)
    FakeInferenceClient.status = None
    gen = ImageGenerator(replace(settings, image_provider="huggingface", hf_token="hf_x"))
    image = asyncio.run(gen.generate("prompt"))
    assert image["mime_type"] == "image/png"
    assert FakeInferenceClient.seen == {"provider": "auto", "api_key": "hf_x",
                                        "model": "black-forest-labs/FLUX.1-schnell", "size": (1024, 576)}


@pytest.mark.parametrize("status, phrase", [(401, "Inference Providers"), (402, "credits")])
def test_huggingface_errors_are_actionable(settings, monkeypatch, status, phrase):
    monkeypatch.setattr("huggingface_hub.InferenceClient", FakeInferenceClient)
    FakeInferenceClient.status = status
    gen = ImageGenerator(replace(settings, image_provider="huggingface", hf_token="hf_x"))
    with pytest.raises(ImageGenerationError, match=phrase):
        asyncio.run(gen.generate("prompt"))
