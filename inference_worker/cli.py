"""CLI entry point — starts web dashboard + optional Tkinter GUI."""

import argparse
import logging
import os
import sys
import threading

# With PyInstaller --noconsole, sys.stdout/stderr can be None and uvicorn's formatter fails.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")


def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _has_display() -> bool:
    """Check if a graphical display is available."""
    if sys.platform == "win32":
        if getattr(sys, "frozen", False):
            return True
        return os.environ.get("SESSIONNAME") is not None or os.environ.get("DISPLAY") is not None
    if sys.platform == "darwin":
        # macOS uses Quartz, not X11 — if we're not in an SSH session, assume display
        return not os.environ.get("SSH_CONNECTION")
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _apply_cli_overrides(args):
    """Push CLI flag values into Settings before the web app reads them."""
    from .config import Settings
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


def main():
    parser = argparse.ArgumentParser(
        prog="grid-inference-worker",
        description="Turn-key text inference worker for AI Power Grid",
    )
    parser.add_argument("--headless", action="store_true",
                        help="Skip the desktop GUI window")
    parser.add_argument("--model", metavar="NAME",
                        help="Model name (e.g. llama3.2:3b)")
    parser.add_argument("--backend-url", metavar="URL",
                        help="Backend URL (e.g. http://127.0.0.1:11434)")
    parser.add_argument("--api-key", metavar="KEY",
                        help="Grid API key")
    parser.add_argument("--worker-name", metavar="NAME",
                        help="Worker name on the grid")
    parser.add_argument("--port", type=int, default=7861, metavar="PORT",
                        help="Web dashboard port (default: 7861)")
    parser.add_argument("--install-service", action="store_true",
                        help="Install as a system service (systemd/launchd/Windows startup)")
    parser.add_argument("--uninstall-service", action="store_true",
                        help="Remove the system service")
    parser.add_argument("--service-status", action="store_true",
                        help="Check if the service is installed and running")
    args = parser.parse_args()

    _setup_logging()

    # Service commands (no worker, just install/remove/status)
    if args.service_status:
        from . import service
        service.status()
        return
    if args.install_service:
        from .env_utils import is_configured
        from . import service
        if not is_configured():
            print("  Error: configure the worker first (run grid-inference-worker to set up).")
            sys.exit(1)
        service.install(verbose=True)
        return
    if args.uninstall_service:
        from . import service
        service.uninstall(verbose=True)
        return

    # Apply CLI overrides
    _apply_cli_overrides(args)

    host = "0.0.0.0"
    port = args.port
    url = f"http://localhost:{port}"

    show_gui = _has_display() and not args.headless

    # Start web server in background thread
    def run_server():
        import uvicorn
        from .web.app import app
        uvicorn.run(app, host=host, port=port, log_level="warning")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    if show_gui:
        from . import gui
        gui.run(url)
    else:
        # Headless: print URL and block until Ctrl+C
        logger = logging.getLogger(__name__)
        logger.info(f"Dashboard: {url}")
        try:
            server_thread.join()
        except KeyboardInterrupt:
            print("\n  Shutting down...")


if __name__ == "__main__":
    main()
