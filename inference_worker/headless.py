"""Headless mode — interactive quick setup + background worker loop."""

import asyncio
import getpass
import sys

from .config import Settings
from .env_utils import ENV_PATH, is_configured, write_env, reload_settings
from .worker import ENLISTMENT_PROMPT, strip_thinking_tags
from . import service


def quick_setup() -> dict:
    """Interactive terminal setup. Returns config dict ready for .env."""
    import httpx
    from .detect_backends import detect_backends, check_backend_url, list_models_for_backend

    print()
    print("  +- Grid Inference Worker -----------------------+")
    print("  | No configuration found. Starting quick setup.  |")
    print("  +------------------------------------------------+")
    print()

    config = {}

    # --- Backend detection ---
    print("  Scanning for backends...", end=" ", flush=True)
    detection = detect_backends()

    if detection.found:
        b = detection.backends[0]
        print(f"found {b.name} @ {b.url}", end="")
        if b.version:
            print(f" (v{b.version})", end="")
        print()

        config["BACKEND_TYPE"] = "ollama" if b.engine == "ollama" else "openai"
        if b.engine == "ollama":
            config["OLLAMA_URL"] = b.url
        else:
            config["OPENAI_URL"] = b.url + "/v1"
        backend_url = b.url
        engine = b.engine
        models = b.models

        if len(detection.backends) > 1:
            print()
            print("  Detected backends:")
            for i, be in enumerate(detection.backends, 1):
                tag = f" (v{be.version})" if be.version else ""
                print(f"    [{i}] {be.name} @ {be.url}{tag}")
            choice = input(f"\n  Select backend [1]: ").strip()
            if choice and choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(detection.backends):
                    b = detection.backends[idx]
                    config["BACKEND_TYPE"] = "ollama" if b.engine == "ollama" else "openai"
                    if b.engine == "ollama":
                        config["OLLAMA_URL"] = b.url
                    else:
                        config["OPENAI_URL"] = b.url + "/v1"
                    backend_url = b.url
                    engine = b.engine
                    models = b.models
    else:
        print("none found.")
        print()
        backend_url = input("  Backend URL: ").strip()
        if not backend_url:
            print("  No backend URL provided. Exiting.")
            sys.exit(1)

        print("  Checking...", end=" ", flush=True)
        info = asyncio.run(check_backend_url(backend_url))

        if info.get("auth_required"):
            print("authentication required.")
            api_key = getpass.getpass("  API key for this backend: ")
            config["OPENAI_API_KEY"] = api_key
            info = asyncio.run(check_backend_url(backend_url, api_key=api_key))

        if info.get("reachable"):
            name = info.get("name", "Unknown")
            print(f"connected ({name})")
        else:
            print("could not connect.")
            cont = input("  Continue anyway? [y/N]: ").strip().lower()
            if cont != "y":
                sys.exit(1)

        engine = info.get("engine", "openai-compat")
        if engine == "ollama":
            config["BACKEND_TYPE"] = "ollama"
            config["OLLAMA_URL"] = backend_url
        else:
            config["BACKEND_TYPE"] = "openai"
            config["OPENAI_URL"] = backend_url.rstrip("/") + "/v1"

        models = info.get("models", [])
        if not models:
            models = asyncio.run(list_models_for_backend(
                backend_url, engine, api_key=config.get("OPENAI_API_KEY", ""),
            ))

    # --- Model selection ---
    print()
    if models:
        print("  Available models:")
        for i, m in enumerate(models[:20], 1):
            print(f"    [{i}] {m}")
        if len(models) > 20:
            print(f"    ... and {len(models) - 20} more")
        print(f"    [{len(models[:20]) + 1}] Enter model name manually")
        print()
        choice = input(f"  Select model [1]: ").strip()
        if not choice:
            model = models[0]
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models[:20]):
                model = models[idx]
            else:
                model = input("  Model name: ").strip()
        else:
            model = choice
    else:
        model = input("  Model name (e.g. llama3.2:3b): ").strip()

    if not model:
        print("  No model selected. Exiting.")
        sys.exit(1)
    config["MODEL_NAME"] = model
    config["GRID_MODEL_NAME"] = f"grid/{model}"

    # --- Grid API key ---
    api_key = getpass.getpass("  Grid API key: ")
    if not api_key:
        print("  No API key provided. Exiting.")
        sys.exit(1)
    config["GRID_API_KEY"] = api_key

    # --- Worker name ---
    worker_name = input("  Worker name [Text-Inference-Worker]: ").strip()
    config["GRID_WORKER_NAME"] = worker_name or "Text-Inference-Worker"

    # --- Enlistment test ---
    print()
    print(f"  Enlisting {model}...", flush=True)

    prompt = ENLISTMENT_PROMPT.format(model=model)

    chat_url = f"{backend_url.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {config['OPENAI_API_KEY']}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 80,
        "temperature": 0.8,
    }
    if engine == "ollama":
        payload["think"] = False

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(chat_url, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                choice = data.get("choices", [{}])[0]
                reply = (choice.get("message", {}).get("content") or "").strip()
                reply = strip_thinking_tags(reply)
                if choice.get("finish_reason") == "length":
                    reply += " ..."
                import textwrap
                wrapped = textwrap.fill(reply, width=68, initial_indent='  "', subsequent_indent='   ')
                print(f'{wrapped}"')
            else:
                print(f"  Warning: backend returned HTTP {resp.status_code} (worker may still work)")
    except Exception as e:
        print(f"  Warning: enlistment test failed ({e})")

    # --- Save ---
    print()
    write_env(config)
    print(f"  Config saved to {ENV_PATH}")

    # --- Offer service installation ---
    print()
    print("  Install as system service?")
    print("    [1] Yes — start on boot, run in background (Recommended)")
    print("    [2] No — run in foreground now")
    choice = input("  [1]: ").strip()
    if choice != "2":
        print()
        service.install(verbose=True)
        config["_service_installed"] = True
    print()

    return config


def run(args):
    """Run worker in headless mode (no GUI, no web server)."""
    # Apply CLI flag overrides
    if args.api_key:
        Settings.GRID_API_KEY = args.api_key
    if args.model:
        Settings.MODEL_NAME = args.model
        if not Settings.GRID_MODEL_NAME:
            Settings.GRID_MODEL_NAME = f"grid/{args.model}"
    if args.backend_url:
        url = args.backend_url.rstrip("/")
        try:
            import httpx
            r = httpx.get(f"{url}/api/version", timeout=2)
            if r.status_code == 200:
                Settings.BACKEND_TYPE = "ollama"
                Settings.OLLAMA_URL = url
            else:
                raise Exception()
        except Exception:
            Settings.BACKEND_TYPE = "openai"
            Settings.OPENAI_URL = url + "/v1"
    if args.worker_name:
        Settings.GRID_WORKER_NAME = args.worker_name

    if not is_configured():
        if args.no_setup:
            print("Error: GRID_API_KEY and MODEL_NAME are required.")
            print("Set them via env vars, .env, or CLI flags. Run without --no-setup for interactive setup.")
            sys.exit(1)
        config = quick_setup()
        reload_settings(config)

        if config.get("_service_installed"):
            return

    print("  Starting worker...")
    print()

    from .worker import TextWorker

    worker = TextWorker()

    async def _run():
        try:
            await worker.run()
        except asyncio.CancelledError:
            pass
        finally:
            await worker.cleanup()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n  Shutting down...")
