# One-shot setup for Windows (PowerShell):
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# 1. creates .venv and installs requirements.txt
# 2. makes sure Ollama is installed and running (offers to install it with winget)
# 3. downloads the Qwen chat model, the embedding model and the reranker
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# --- 1. Python environment -----------------------------------------------------
$python = $null
foreach ($candidate in @("py -3", "python")) {
    $exe, $args0 = $candidate.Split(" ", 2)
    if (Get-Command $exe -ErrorAction SilentlyContinue) { $python = $candidate; break }
}
if (-not $python) { throw "Python 3.10+ is required: https://www.python.org/downloads/windows/" }

if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating virtual environment (.venv)"
    Invoke-Expression "$python -m venv .venv"
}
$venvPython = Join-Path ".venv" "Scripts\python.exe"
& $venvPython -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 'Python 3.10+ is required')"
Write-Host "==> Installing Python dependencies"
& $venvPython -m pip install --upgrade pip | Out-Null
& $venvPython -m pip install -r requirements.txt

# --- 2. Ollama -------------------------------------------------------------------
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "==> Ollama is not installed (it serves the local Qwen model)."
    $answer = Read-Host "Install it now with winget (winget install Ollama.Ollama)? [y/N]"
    if ($answer -match "^[yY]") {
        winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")
    } else {
        Write-Host "Download it from https://ollama.com/download/windows, then re-run this script."
        exit 1
    }
}

$ollamaUrl = if ($env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL } else { "http://127.0.0.1:11434" }
try { Invoke-RestMethod "$ollamaUrl/api/version" | Out-Null }
catch {
    Write-Host "==> Starting Ollama in the background"
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30; $i++) {
        try { Invoke-RestMethod "$ollamaUrl/api/version" | Out-Null; break } catch { Start-Sleep 1 }
    }
}

# --- 3. Models ---------------------------------------------------------------------
Write-Host "==> Downloading models (first run: ~3.5 GB for Qwen 3.5 4B + embeddings + reranker)"
& $venvPython -m rag_app.setup_models

Write-Host ""
Write-Host "All set! Start the app with:"
Write-Host "    .venv\Scripts\python.exe main.py"
Write-Host "then open http://127.0.0.1:8080"
