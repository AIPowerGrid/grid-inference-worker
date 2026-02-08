import os
import logging
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ..config import Settings
from ..ollama_detect import (
    detect_backends,
    check_backend_url,
    list_models_for_backend,
    get_model_context_length,
    install_ollama,
    pull_ollama_model,
    get_platform,
)
from .app import app, templates, worker_state, start_worker, stop_worker

logger = logging.getLogger(__name__)

ENV_PATH = Path.cwd() / ".env"


# ---------------------------------------------------------------------------
# Middleware: redirect to setup if not configured
# ---------------------------------------------------------------------------
@app.middleware("http")
async def setup_guard(request: Request, call_next):
    path = request.url.path
    if (
        path.startswith("/static")
        or path.startswith("/api/")
        or path.startswith("/setup")
    ):
        return await call_next(request)
    if not worker_state["setup_complete"]:
        return RedirectResponse("/setup", status_code=303)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Setup wizard
# ---------------------------------------------------------------------------
@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    detection = detect_backends()
    return templates.TemplateResponse("setup.html", {
        "request": request,
        "detection": detection,
        "platform": get_platform(),
    })


@app.post("/api/setup/detect")
async def api_detect():
    """Scan all known ports for running inference engines."""
    detection = detect_backends()
    return {
        "found": detection.found,
        "ollama_binary": detection.ollama_binary,
        "ollama_version": detection.ollama_version,
        "backends": [
            {
                "engine": b.engine,
                "name": b.name,
                "url": b.url,
                "models": b.models,
                "version": b.version,
                "api_type": b.api_type,
            }
            for b in detection.backends
        ],
    }


@app.post("/api/setup/check-url")
async def api_check_url(request: Request):
    """Probe a specific URL and identify the engine."""
    body = await request.json()
    url = body.get("url", "")
    info = await check_backend_url(url)
    return info


@app.post("/api/setup/install-ollama")
async def api_install_ollama():
    """Install Ollama using the official install script."""
    result = install_ollama()
    return result


@app.post("/api/setup/pull-model")
async def api_pull_model(request: Request):
    """Pull an Ollama model."""
    body = await request.json()
    url = body.get("url", Settings.OLLAMA_URL)
    model = body.get("model", "")
    if not model:
        return {"ok": False, "error": "No model name provided"}
    result = await pull_ollama_model(url, model)
    return result


@app.post("/api/setup/context-length")
async def api_context_length(request: Request):
    """Detect model context length from the backend."""
    body = await request.json()
    url = body.get("url", Settings.OLLAMA_URL)
    engine = body.get("engine")
    model = body.get("model", "")
    result = await get_model_context_length(url, engine, model)
    return result


@app.post("/api/setup/list-models")
async def api_list_models(request: Request):
    """List models available on any backend."""
    body = await request.json()
    url = body.get("url", Settings.OLLAMA_URL)
    engine = body.get("engine")
    models = await list_models_for_backend(url, engine)
    return {"models": models}


@app.post("/api/setup/complete")
async def api_complete_setup(request: Request):
    """Save config and start the worker."""
    form = await request.json()

    # Build .env content, preserving any existing keys
    env_lines = _read_existing_env()
    for key, value in form.items():
        if value is not None and value != "":
            env_lines[key] = value

    # Write .env
    content = "\n".join(f"{k}={v}" for k, v in env_lines.items()) + "\n"
    ENV_PATH.write_text(content)

    # Reload settings in memory
    _reload_settings(form)

    worker_state["setup_complete"] = True

    # Start worker
    if Settings.GRID_API_KEY and Settings.MODEL_NAME:
        await start_worker()

    logger.info("Setup complete. Worker starting.")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "worker_running": worker_state["running"],
        "worker_error": worker_state.get("error"),
    })


@app.get("/api/status")
async def api_status():
    return {
        "worker_running": worker_state["running"],
        "worker_error": worker_state.get("error"),
        "config": {
            "has_api_key": bool(Settings.GRID_API_KEY),
            "worker_name": Settings.GRID_WORKER_NAME,
            "backend_type": Settings.BACKEND_TYPE,
            "ollama_url": Settings.OLLAMA_URL,
            "model_name": Settings.MODEL_NAME,
            "grid_model_name": Settings.GRID_MODEL_NAME,
            "max_threads": Settings.MAX_THREADS,
            "max_length": Settings.MAX_LENGTH,
            "max_context_length": Settings.MAX_CONTEXT_LENGTH,
            "nsfw": Settings.NSFW,
            "wallet_address": Settings.WALLET_ADDRESS,
        },
    }


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "settings": {
            "GRID_API_KEY": Settings.GRID_API_KEY,
            "GRID_WORKER_NAME": Settings.GRID_WORKER_NAME,
            "BACKEND_TYPE": Settings.BACKEND_TYPE,
            "OLLAMA_URL": Settings.OLLAMA_URL,
            "OPENAI_URL": Settings.OPENAI_URL,
            "OPENAI_API_KEY": Settings.OPENAI_API_KEY,
            "MODEL_NAME": Settings.MODEL_NAME,
            "GRID_MODEL_NAME": Settings.GRID_MODEL_NAME,
            "GRID_NSFW": str(Settings.NSFW).lower(),
            "GRID_MAX_THREADS": str(Settings.MAX_THREADS),
            "GRID_MAX_LENGTH": str(Settings.MAX_LENGTH),
            "GRID_MAX_CONTEXT_LENGTH": str(Settings.MAX_CONTEXT_LENGTH),
            "WALLET_ADDRESS": Settings.WALLET_ADDRESS,
        },
    })


@app.post("/api/settings")
async def save_settings(request: Request):
    """Save settings to .env and update in-memory config."""
    form = await request.json()

    env_lines = _read_existing_env()
    for key, value in form.items():
        if value is not None and value != "":
            env_lines[key] = value
        elif key in env_lines:
            del env_lines[key]

    content = "\n".join(f"{k}={v}" for k, v in env_lines.items()) + "\n"
    ENV_PATH.write_text(content)
    _reload_settings(form)

    logger.info(f"Settings saved to {ENV_PATH}")
    return {"ok": True, "message": "Restart worker to apply all changes."}


@app.post("/api/worker/restart")
async def restart_worker():
    """Stop and restart the worker with current config."""
    await stop_worker()
    await start_worker()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_existing_env() -> dict:
    """Read existing .env into an ordered dict."""
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _reload_settings(form: dict):
    """Update Settings class attributes from form data."""
    if "GRID_API_KEY" in form:
        Settings.GRID_API_KEY = form["GRID_API_KEY"]
    if "GRID_WORKER_NAME" in form:
        Settings.GRID_WORKER_NAME = form["GRID_WORKER_NAME"]
    if "BACKEND_TYPE" in form:
        Settings.BACKEND_TYPE = form["BACKEND_TYPE"]
    if "OLLAMA_URL" in form:
        Settings.OLLAMA_URL = form["OLLAMA_URL"]
    if "OPENAI_URL" in form:
        Settings.OPENAI_URL = form["OPENAI_URL"]
    if "OPENAI_API_KEY" in form:
        Settings.OPENAI_API_KEY = form["OPENAI_API_KEY"]
    if "MODEL_NAME" in form:
        Settings.MODEL_NAME = form["MODEL_NAME"]
    if "GRID_MODEL_NAME" in form:
        Settings.GRID_MODEL_NAME = form["GRID_MODEL_NAME"]
    if "GRID_NSFW" in form:
        Settings.NSFW = form["GRID_NSFW"].lower() == "true"
    if "GRID_MAX_THREADS" in form:
        Settings.MAX_THREADS = int(form["GRID_MAX_THREADS"])
    if "GRID_MAX_LENGTH" in form:
        Settings.MAX_LENGTH = int(form["GRID_MAX_LENGTH"])
    if "GRID_MAX_CONTEXT_LENGTH" in form:
        Settings.MAX_CONTEXT_LENGTH = int(form["GRID_MAX_CONTEXT_LENGTH"])
    if "WALLET_ADDRESS" in form:
        Settings.WALLET_ADDRESS = form["WALLET_ADDRESS"]
