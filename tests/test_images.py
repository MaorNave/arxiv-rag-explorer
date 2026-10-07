"""Free image providers and their fallback chain, tested offline with httpx.MockTransport."""

import asyncio
import base64
import json
from dataclasses import replace

import httpx
import pytest

from rag_app.images import IMAGEGEN_URL, ImageGenerationError, ImageGenerator

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
CLOUDFLARE = {"cloudflare_account_id": "acc", "cloudflare_api_token": "cf_token"}


def generator(settings, handler, local=False, **overrides):
    gen = ImageGenerator(replace(settings, **overrides), transport=httpx.MockTransport(handler))
    gen.local_url = IMAGEGEN_URL if local else None
    return gen


def ok_pollinations(request):
    assert request.url.host == "image.pollinations.ai"
    return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})


def test_auto_chain_prefers_local_then_free_apis(settings):
    auto = replace(settings, image_provider="auto")
    assert ImageGenerator(auto).chain() == ["pollinations"]
    assert ImageGenerator(replace(auto, **CLOUDFLARE)).chain() == ["cloudflare", "pollinations"]
    gen = ImageGenerator(replace(auto, **CLOUDFLARE))
    gen.local_url = IMAGEGEN_URL
    assert gen.chain() == ["ollama", "cloudflare", "pollinations"] and gen.runs_after_answer
    assert gen.describe()["chain"] == ["Ollama (local)", "Cloudflare Workers AI", "Pollinations.ai"]


def test_explicit_and_disabled_providers(settings):
    assert not ImageGenerator(replace(settings, image_provider="none")).available
    assert "CLOUDFLARE_ACCOUNT_ID" in ImageGenerator(replace(settings, image_provider="cloudflare")).unavailable_reason
    assert "Unknown" in ImageGenerator(replace(settings, image_provider="openai")).unavailable_reason
    assert ImageGenerator(replace(settings, image_provider="pollinations")).chain() == ["pollinations"]


def test_local_candidates(settings):
    gen = ImageGenerator(settings)
    assert gen.local_candidates() == [IMAGEGEN_URL, settings.ollama_base_url]  # dedicated 0.32.5 server first
    gen.unsupported_urls.add(settings.ollama_base_url)
    assert gen.local_candidates() == [IMAGEGEN_URL]
    explicit = ImageGenerator(replace(settings, ollama_image_base_url="http://gpu-box:11434"))
    assert explicit.local_candidates() == ["http://gpu-box:11434"]


def test_local_ollama_generates_after_the_answer(settings):
    def handler(request):
        body = json.loads(request.content)
        assert str(request.url) == f"{IMAGEGEN_URL}/api/generate" and body["model"] == "x/flux2-klein:4b"
        assert body["stream"] is False and body["keep_alive"] == 0 and (body["width"], body["height"]) == (1024, 576)
        return httpx.Response(200, json={"image": base64.b64encode(PNG).decode(), "done": True})

    gen = generator(settings, handler, local=True, image_provider="auto")
    image = asyncio.run(gen.generate("a lighthouse"))
    assert image["provider"] == "ollama" and image["mime_type"] == "image/png"
    assert (settings.images_dir / image["url"].split("/")[-1]).read_bytes() == PNG


def test_unsupported_ollama_falls_back_and_is_skipped_afterwards(settings):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.path == "/api/generate":
            return httpx.Response(400, json={"error": "image generation models are not currently supported"})
        return ok_pollinations(request)

    gen = generator(settings, handler, local=True, image_provider="auto")
    assert gen.runs_after_answer
    assert asyncio.run(gen.generate("prompt"))["provider"] == "pollinations"
    assert gen.chain() == ["pollinations"] and not gen.runs_after_answer  # remembered for later requests
    asyncio.run(gen.generate("prompt"))
    assert calls == ["127.0.0.1", "image.pollinations.ai", "image.pollinations.ai"]


def test_cloudflare_workers_ai(settings):
    def handler(request):
        assert request.url.path == "/client/v4/accounts/acc/ai/run/@cf/black-forest-labs/flux-1-schnell"
        assert request.headers["Authorization"] == "Bearer cf_token"
        assert json.loads(request.content) == {"prompt": "prompt", "steps": 4}
        return httpx.Response(200, json={"result": {"image": base64.b64encode(JPEG).decode()}, "success": True})

    image = asyncio.run(generator(settings, handler, image_provider="auto", **CLOUDFLARE).generate("prompt"))
    assert image["provider"] == "cloudflare" and image["mime_type"] == "image/jpeg"


@pytest.mark.parametrize("status, body, phrase", [
    (401, {"errors": [{"message": "Authentication error"}]}, "API token"),
    (429, {"errors": [{"message": "Too many requests"}]}, "free daily allowance"),
    (400, {"errors": [{"message": "you have used up your daily free allocation of 10,000 neurons"}]}, "free daily allowance"),
])
def test_cloudflare_errors_fall_back_with_clear_messages(settings, status, body, phrase):
    def handler(request):
        if request.url.host == "api.cloudflare.com":
            return httpx.Response(status, json=body)
        return ok_pollinations(request)

    gen = generator(settings, handler, image_provider="auto", **CLOUDFLARE)
    assert asyncio.run(gen.generate("prompt"))["provider"] == "pollinations"  # fallback worked

    forced = generator(settings, handler, image_provider="cloudflare", **CLOUDFLARE)
    with pytest.raises(ImageGenerationError, match=phrase):
        asyncio.run(forced.generate("prompt"))


@pytest.mark.parametrize("status, phrase", [(402, "rate-limited"), (429, "rate-limited"), (500, "best-effort")])
def test_keyless_pollinations_errors_are_actionable(settings, status, phrase):
    gen = generator(settings, lambda r: httpx.Response(status, json={"error": "x"}), image_provider="pollinations")
    with pytest.raises(ImageGenerationError, match=phrase):
        asyncio.run(gen.generate("prompt"))


def test_pollinations_with_free_key_uses_the_authenticated_api(settings):
    def handler(request):
        assert request.url.host == "gen.pollinations.ai"
        assert request.headers["Authorization"] == "Bearer sk_test" and request.url.params["model"] == "flux"
        return httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"})

    gen = generator(settings, handler, image_provider="auto", pollinations_token="sk_test")
    assert asyncio.run(gen.generate("prompt"))["url"].endswith(".jpg")
    assert gen.describe()["keyless"] is False


def test_all_providers_failing_reports_every_reason(settings):
    def handler(request):
        if request.url.path == "/api/generate":
            return httpx.Response(500, json={"error": "insufficient memory for image generation"})
        return httpx.Response(429, json={"error": "slow down"})

    gen = generator(settings, handler, local=True, image_provider="auto")
    with pytest.raises(ImageGenerationError) as err:
        asyncio.run(gen.generate("prompt"))
    assert "Ollama (local)" in str(err.value) and "Pollinations.ai" in str(err.value)
