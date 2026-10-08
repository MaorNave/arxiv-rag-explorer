"""Ollama managed by the app itself, so `python main.py` needs no outside installation.

* No Ollama running on this computer? When OLLAMA_BASE_URL points at this machine, the app
  downloads Ollama into STORAGE_DIR/runtime and runs it as a child process: it starts with the
  app and stops with the app. Models that a regular Ollama installation already has
  (~/.ollama/models) are reused as copy-on-write clones or hard links instead of being
  downloaded again.
* Local images (macOS): Ollama removed its experimental image generation in 0.32.6, and its
  release notes say to keep using 0.32.5 for it. The private runtime is pinned to 0.32.5, so it
  can serve FLUX.2 klein. When the regular Ollama is newer, the app runs the private copy next to
  it, on LOCAL_IMAGE_PORT, for images only.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .config import Settings

log = logging.getLogger(__name__)

RUNTIME_VERSION = "0.32.5"  # last Ollama release with (experimental) image generation
_RELEASE_URL = "https://github.com/ollama/ollama/releases/download/v{version}/{asset}"
Progress = Callable[["float | None", str], None]


def platform_asset() -> str | None:
    """Name of the Ollama release archive for this computer."""
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "amd64"
    return {
        "Darwin": "ollama-darwin.tgz",
        "Linux": f"ollama-linux-{arch}.tar.zst",
        "Windows": f"ollama-windows-{arch}.zip",
    }.get(platform.system())


def system_models_dir() -> Path:
    """Where a regular Ollama installation keeps its models."""
    return Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models").expanduser()


def is_local_url(url: str) -> bool:
    return (urlparse(url).hostname or "") in ("127.0.0.1", "localhost", "0.0.0.0", "::1")


def version_tuple(version: str) -> tuple[int, ...]:
    numbers = []
    for part in version.split("-")[0].split("."):
        if not part.isdigit():
            break
        numbers.append(int(part))
    return tuple(numbers)


def supports_image_generation(version: str | None) -> bool:
    """Experimental image generation shipped in Ollama 0.14.0 and was removed in 0.32.6."""
    return bool(version) and (0, 14, 0) <= version_tuple(version) < (0, 32, 6)


def total_memory_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (AttributeError, OSError, ValueError):
        return None


def manifest_relpath(model: str) -> Path:
    name, _, tag = model.partition(":")
    parts = name.split("/")
    if len(parts) == 1:
        parts = ["registry.ollama.ai", "library", parts[0]]
    elif len(parts) == 2:
        parts = ["registry.ollama.ai", *parts]
    return Path("manifests", *parts, tag or "latest")


def _clone_file(src: Path, dst: Path) -> None:
    """Copy-on-write clone (APFS), else hard link, else a regular copy."""
    if sys.platform == "darwin" and subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True).returncode == 0:
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def import_model(model: str, source: Path, target: Path) -> bool:
    """Reuse a model from another Ollama model store without downloading it again."""
    rel = manifest_relpath(model)
    if (target / rel).exists():
        return True
    source_manifest = source / rel
    if not source_manifest.exists():
        return False
    data = json.loads(source_manifest.read_text())
    digests = [data["config"]["digest"], *(layer["digest"] for layer in data["layers"])]
    (target / "blobs").mkdir(parents=True, exist_ok=True)
    for digest in dict.fromkeys(digests):
        name = digest.replace(":", "-")
        if not (target / "blobs" / name).exists():
            if not (source / "blobs" / name).exists():
                return False  # incomplete source store
            _clone_file(source / "blobs" / name, target / "blobs" / name)
    (target / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_manifest, target / rel)  # manifest last: the import is complete
    log.info("Reused %s from %s (no download)", model, source)
    return True


def _extract(archive: Path, asset: str, target: Path) -> None:
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        return
    safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    if asset.endswith(".tar.zst"):
        import zstandard  # only needed for the Linux builds

        with (
            archive.open("rb") as fh,
            zstandard.ZstdDecompressor().stream_reader(fh) as reader,
            tarfile.open(fileobj=reader, mode="r|") as tar,
        ):
            tar.extractall(target, **safe)
    else:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(target, **safe)


class ManagedOllama:
    """One `ollama serve` process owned by the app."""

    def __init__(self, runtime_dir: Path, port: int, models_dir: Path, version: str = RUNTIME_VERSION,
                 transport: httpx.BaseTransport | None = None):
        self.version = version
        self.install_dir = runtime_dir / f"ollama-{version}"
        self.models_dir = models_dir
        self.port = port
        self.log_path = runtime_dir / f"ollama-{port}.log"
        self.pid_path = runtime_dir / f"ollama-{port}.pid"
        self.process: subprocess.Popen | None = None
        self.adopted_pid: int | None = None
        self._cancel = threading.Event()  # set by stop(): aborts a download or start still in progress
        self._lock = threading.Lock()  # stop() can run on another thread while start() spawns the server
        self._transport = transport  # injectable for offline tests

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def binary(self) -> Path | None:
        names = ("ollama.exe", "ollama") if sys.platform == "win32" else ("ollama",)
        for folder in (self.install_dir, self.install_dir / "bin"):
            for name in names:
                if (folder / name).is_file():
                    return folder / name
        return None

    def install(self, progress: Progress | None = None) -> None:
        """Download and unpack Ollama (once)."""
        if self.binary:
            return
        asset = platform_asset()
        if asset is None:
            raise RuntimeError(f"No Ollama build available for {platform.system()} {platform.machine()}")
        url = _RELEASE_URL.format(version=self.version, asset=asset)
        self.install_dir.parent.mkdir(parents=True, exist_ok=True)
        archive = self.install_dir.parent / f"{asset}.part"
        log.info("Downloading Ollama %s (%s)", self.version, asset)
        timeout = httpx.Timeout(60.0, read=300.0)
        with (
            httpx.Client(transport=self._transport, follow_redirects=True, timeout=timeout) as client,
            client.stream("GET", url) as response,
        ):
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            done = 0
            with archive.open("wb") as fh:
                for block in response.iter_bytes(1 << 20):
                    if self._cancel.is_set():
                        raise RuntimeError("Ollama download cancelled (the app is shutting down)")
                    fh.write(block)
                    done += len(block)
                    if progress:
                        percent = round(100 * done / total, 1) if total else None
                        progress(percent, f"downloading Ollama {self.version} ({done // 2**20} MB)")
        if progress:
            progress(None, f"unpacking Ollama {self.version}")
        staging = self.install_dir.with_name(self.install_dir.name + ".staging")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        _extract(archive, asset, staging)
        archive.unlink(missing_ok=True)
        shutil.rmtree(self.install_dir, ignore_errors=True)
        staging.rename(self.install_dir)
        binary = self.binary
        if binary is None:
            raise RuntimeError("The Ollama download did not contain the ollama program")
        binary.chmod(binary.stat().st_mode | 0o111)

    def is_up(self) -> bool:
        try:
            with httpx.Client(transport=self._transport, timeout=2.0) as client:
                return client.get(f"{self.url}/api/version").status_code == 200
        except httpx.HTTPError:
            return False

    def start(self, timeout_s: float = 90.0) -> None:
        if self.is_up():  # e.g. left running by a previous run that crashed: reuse and own it
            if self.pid_path.exists():
                self.adopted_pid = int(self.pid_path.read_text().strip() or 0) or None
            log.info("Reusing the Ollama server already running at %s", self.url)
            return
        binary = self.binary
        if binary is None:
            raise RuntimeError("Ollama is not installed yet")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "OLLAMA_HOST": f"127.0.0.1:{self.port}",
            "OLLAMA_MODELS": str(self.models_dir),
            "OLLAMA_NOPRUNE": "1",  # never delete model files
        }
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        with self._lock:
            if self._cancel.is_set():  # the app shut down while this was being set up
                raise RuntimeError("Ollama start cancelled (the app is shutting down)")
            with self.log_path.open("ab") as log_file:
                process = self.process = subprocess.Popen(
                    [str(binary), "serve"], env=env, cwd=str(binary.parent),
                    stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, creationflags=flags,
                )
            self.pid_path.write_text(str(process.pid))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Ollama exited during startup; see {self.log_path}")
            if self.is_up():
                log.info("Started Ollama %s at %s (models: %s)", self.version, self.url, self.models_dir)
                return
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"Ollama did not start within {timeout_s:.0f}s; see {self.log_path}")

    def stop(self) -> None:
        with self._lock:
            self._cancel.set()
            process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            log.info("Stopped the app's Ollama at %s", self.url)
        elif self.adopted_pid:
            try:
                os.kill(self.adopted_pid, signal.SIGTERM)
            except OSError:
                pass
        self.adopted_pid = None
        self.pid_path.unlink(missing_ok=True)


class RuntimeManager:
    """Decides which Ollama servers the app has to run itself."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.dir = settings.storage_dir / "runtime"
        self.models_dir = self.dir / "models"
        self.main: ManagedOllama | None = None
        self.images: ManagedOllama | None = None

    def can_run_main(self) -> bool:
        s = self.settings
        return s.managed_ollama and is_local_url(s.ollama_base_url) and platform_asset() is not None

    def start_main(self, models: list[str], progress: Progress | None = None) -> None:
        """No Ollama running: install and run the app's own copy on the OLLAMA_BASE_URL port."""
        port = urlparse(self.settings.ollama_base_url).port or 11434
        server = self.main = ManagedOllama(self.dir, port, self.models_dir, self.settings.ollama_runtime_version)
        try:
            server.install(progress)
            for model in models:
                import_model(model, system_models_dir(), self.models_dir)
            server.start()
        except BaseException:
            server.stop()  # a server spawned before the error (e.g. Ctrl+C while waiting) must not outlive it
            self.main = None
            raise

    def start_image_server(self, progress: Progress | None = None) -> ManagedOllama:
        """The regular Ollama is too new for images: run the 0.32.5 copy next to it."""
        server = self.images = ManagedOllama(
            self.dir, self.settings.local_image_port, self.models_dir, self.settings.ollama_runtime_version
        )
        try:
            server.install(progress)
            import_model(self.settings.ollama_image_model, system_models_dir(), self.models_dir)
            server.start()
        except BaseException:
            server.stop()
            self.images = None
            raise
        return server

    def stop(self) -> None:
        for server in (self.images, self.main):
            if server is not None:
                server.stop()
