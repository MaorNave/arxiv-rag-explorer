"""The app-managed Ollama runtime, tested offline (fake archive, fake `ollama serve`)."""

import io
import json
import socket
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from rag_app import runtime
from rag_app.runtime import (
    ManagedOllama,
    RuntimeManager,
    import_model,
    is_local_url,
    manifest_relpath,
    supports_image_generation,
)

FAKE_SERVER = """#!{python}
import http.server, os, sys
if sys.argv[1:] != ["serve"]:
    sys.exit(2)
host, port = os.environ["OLLAMA_HOST"].split(":")
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(b'{{"version": "0.32.5"}}')
    def log_message(self, *args):
        pass
http.server.HTTPServer((host, int(port)), Handler).serve_forever()
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake_archive() -> bytes:
    buffer = io.BytesIO()
    script = FAKE_SERVER.format(python=sys.executable).encode()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("ollama")
        info.size, info.mode = len(script), 0o755
        tar.addfile(info, io.BytesIO(script))
    return buffer.getvalue()


def test_helpers():
    assert manifest_relpath("qwen3.5:4b") == Path("manifests/registry.ollama.ai/library/qwen3.5/4b")
    assert manifest_relpath("nomic-embed-text") == Path("manifests/registry.ollama.ai/library/nomic-embed-text/latest")
    assert manifest_relpath("x/flux2-klein:4b") == Path("manifests/registry.ollama.ai/x/flux2-klein/4b")
    assert supports_image_generation("0.32.5") and supports_image_generation("0.14.0")
    assert not supports_image_generation("0.32.6") and not supports_image_generation("0.32.13")
    assert not supports_image_generation("0.13.9") and not supports_image_generation(None)
    assert is_local_url("http://127.0.0.1:11434") and not is_local_url("http://gpu-box:11434")


def test_import_model_reuses_files_without_downloading(tmp_path):
    source, target = tmp_path / "system", tmp_path / "app"
    manifest = source / manifest_relpath("x/flux2-klein:4b")
    manifest.parent.mkdir(parents=True)
    (source / "blobs").mkdir()
    for digest, data in (("sha256:aa", b"config"), ("sha256:bb", b"weights")):
        (source / "blobs" / digest.replace(":", "-")).write_bytes(data)
    manifest.write_text(json.dumps({"config": {"digest": "sha256:aa"}, "layers": [{"digest": "sha256:bb"}]}))

    assert import_model("x/flux2-klein:4b", source, target)
    assert (target / "blobs" / "sha256-bb").read_bytes() == b"weights"
    assert (target / manifest_relpath("x/flux2-klein:4b")).exists()
    assert import_model("x/flux2-klein:4b", source, target)  # already there
    assert not import_model("qwen3.5:4b", source, target)  # not in the source store


@pytest.mark.skipif(sys.platform == "win32", reason="the fake server is a POSIX script")
def test_install_start_and_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "platform_asset", lambda: "ollama-darwin.tgz")
    archive = fake_archive()

    def handler(request):
        assert request.url.path == "/ollama/ollama/releases/download/v0.32.5/ollama-darwin.tgz"
        return httpx.Response(200, content=archive, headers={"content-length": str(len(archive))})

    server = ManagedOllama(tmp_path / "runtime", free_port(), tmp_path / "runtime" / "models",
                           transport=httpx.MockTransport(handler))
    steps = []
    server.install(lambda pct, text: steps.append((pct, text)))
    assert server.binary is not None and steps[-1][1].startswith("unpacking")
    assert any(pct == 100.0 for pct, _ in steps)

    server._transport = None  # talk to the real (fake) server process from here on
    server.start(timeout_s=20)
    try:
        assert server.is_up()
        assert server.pid_path.read_text() == str(server.process.pid)
    finally:
        server.stop()
    assert not server.is_up() and not server.pid_path.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="the fake server is a POSIX script")
def test_shutdown_during_setup_never_leaves_a_server_behind(tmp_path, settings, monkeypatch):
    # The app stops while its setup thread is still on the way to start(): nothing may be spawned.
    server = ManagedOllama(tmp_path / "runtime", free_port(), tmp_path / "models")
    server.install_dir.mkdir(parents=True)
    (server.install_dir / "ollama").write_text("#!/bin/sh\nexit 1\n")
    (server.install_dir / "ollama").chmod(0o755)
    server.stop()
    with pytest.raises(RuntimeError, match="cancelled"):
        server.start()
    assert server.process is None

    # Interrupted after `ollama serve` was spawned (e.g. Ctrl+C while waiting for it): it is stopped.
    spawned = []

    def start_then_interrupt(self, timeout_s=90.0):
        self.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        spawned.append(self.process)
        raise KeyboardInterrupt

    monkeypatch.setattr(ManagedOllama, "install", lambda self, progress=None: None)
    monkeypatch.setattr(ManagedOllama, "start", start_then_interrupt)
    manager = RuntimeManager(settings)
    with pytest.raises(KeyboardInterrupt):
        manager.start_main([])
    assert manager.main is None
    assert spawned[0].wait(timeout=10) is not None  # terminated, not orphaned


def test_runtime_manager_only_manages_local_servers(settings):
    assert RuntimeManager(replace(settings, ollama_base_url="http://127.0.0.1:11434")).can_run_main()
    assert not RuntimeManager(replace(settings, ollama_base_url="http://gpu-box:11434")).can_run_main()
    assert not RuntimeManager(replace(settings, managed_ollama=False)).can_run_main()
