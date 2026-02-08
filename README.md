# Grid Inference Worker

Turn-key text inference worker for [AI Power Grid](https://aipowergrid.io). Bridges between the Grid API and a local text inference backend (Ollama).

## Quick Start

```bash
# Install
pip install -e .

# Run (opens web UI with setup wizard)
grid-inference-worker
```

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

## Backends

**Easy mode (Ollama)** — Install Ollama, pull a model, and go. The worker uses Ollama's OpenAI-compatible API.

**Advanced mode (Coming Soon)** — vLLM, SGLang, LMDeploy for production-grade serving.
