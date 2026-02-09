"""CLI entry point — argparse + routing to headless / GUI / service commands."""

import argparse
import logging
import os
import sys

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
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main():
    parser = argparse.ArgumentParser(
        prog="grid-inference-worker",
        description="Turn-key text inference worker for AI Power Grid",
    )
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI (terminal only)")
    parser.add_argument("--model", metavar="NAME",
                        help="Model name (e.g. llama3.2:3b)")
    parser.add_argument("--backend-url", metavar="URL",
                        help="Backend URL (e.g. http://127.0.0.1:11434)")
    parser.add_argument("--api-key", metavar="KEY",
                        help="Grid API key")
    parser.add_argument("--worker-name", metavar="NAME",
                        help="Worker name on the grid")
    parser.add_argument("--no-setup", action="store_true",
                        help="Fail instead of prompting for missing config")
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

    # Decide mode
    use_headless = (
        args.headless
        or args.no_setup
        or args.model
        or args.api_key
        or args.backend_url
        or not _has_display()
    )

    if use_headless:
        from . import headless
        headless.run(args)
    else:
        from . import gui
        gui.run()


if __name__ == "__main__":
    main()
