# Grid Inference Worker

Turn-key text inference worker for [AI Power Grid](https://aipowergrid.io). Bridges between the Grid API and a local text inference backend (Ollama).

## Quick Start

```bash
# Install
pip install -e .

# Run (opens web UI with setup wizard)
grid-inference-worker
```

**Windows (easy):** In PowerShell from the project folder run:
```powershell
.\scripts\run.ps1
```
This installs the package and starts the worker; your browser will open at the setup wizard.

Open `http://localhost:7861` and the setup wizard will walk you through:
1. Detecting / installing Ollama and selecting a model
2. Entering your Grid API key
3. Launching the worker

## Manual Setup

```bash
cp .env.example .env
# Edit .env with your API key and model
grid-inference-worker
```

## Docker

```bash
cp .env.example .env
# Edit .env
docker compose up -d
```

## Building the EXE (Windows)

To build a standalone executable on Windows:

```powershell
.\scripts\build-exe.ps1
```

Output: `dist\grid-inference-worker.exe` — single file, easy to copy. No Python needed on the target machine; run the exe and open http://localhost:7861.

**Slow startup?** The single-file exe unpacks to a temp folder on every launch (and antivirus may scan it), so the first start can take several seconds. For **fast startup**, build a folder instead:

```powershell
.\scripts\build-exe.ps1 -OneDir
```

Output: `dist\grid-inference-worker\` — run `grid-inference-worker.exe` inside that folder. Copy the whole folder to other machines. No extraction, so it starts quickly.

Requires: Python 3.9+ and pip (e.g. from [python.org](https://www.python.org/downloads/)).

## Backends

**Easy mode (Ollama)** — Install Ollama, pull a model, and go. The worker uses Ollama's OpenAI-compatible API.

**Advanced mode (Coming Soon)** — vLLM, SGLang, LMDeploy for production-grade serving.
