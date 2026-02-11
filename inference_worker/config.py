import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def _config_dir() -> Path:
    """Stable config directory that persists across binary updates."""
    if getattr(sys, "frozen", False):
        if sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        d = base / "grid-inference-worker"
        d.mkdir(parents=True, exist_ok=True)
        # Migrate from old location (next to exe)
        env_new = d / ".env"
        if not env_new.exists():
            env_old = Path(sys.executable).resolve().parent / ".env"
            if env_old.exists():
                import shutil
                shutil.copy2(env_old, env_new)
        return d
    return Path.cwd()


CONFIG_DIR = _config_dir()
ENV_FILE = CONFIG_DIR / ".env"

load_dotenv(ENV_FILE)


class Settings:
    GRID_API_KEY = os.getenv("GRID_API_KEY", "")
    GRID_WORKER_NAME = os.getenv("GRID_WORKER_NAME", "Text-Inference-Worker")
    GRID_API_URL = os.getenv("GRID_API_URL", "https://api.aipowergrid.io/api")
    NSFW = os.getenv("GRID_NSFW", "true").lower() == "true"
    MAX_THREADS = int(os.getenv("GRID_MAX_THREADS", "1"))
    MAX_LENGTH = int(os.getenv("GRID_MAX_LENGTH", "4096"))
    MAX_CONTEXT_LENGTH = int(os.getenv("GRID_MAX_CONTEXT_LENGTH", "4096"))

    # Backend type: "ollama" (easy mode) or "openai" (advanced/custom)
    BACKEND_TYPE = os.getenv("BACKEND_TYPE", "ollama")

    # Ollama settings
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

    # OpenAI-compatible settings (for vllm, sglang, lmdeploy, etc.)
    OPENAI_URL = os.getenv("OPENAI_URL", "http://127.0.0.1:8000/v1")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    # Minimize or disable reasoning on backends that support it (e.g. "low", "none"). Leave unset to not send.
    REASONING_EFFORT = os.getenv("REASONING_EFFORT", "").lower() or None

    # Model to serve (e.g. "llama3.2:3b" for ollama, "meta-llama/..." for openai)
    MODEL_NAME = os.getenv("MODEL_NAME", "")

    # Grid model name (what to advertise to the grid, with domain prefix)
    GRID_MODEL_NAME = os.getenv("GRID_MODEL_NAME", "")

    # Wallet address for rewards (Base chain)
    WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "")

    # Dashboard auth token (auto-generated on first run)
    DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN", "")

    @classmethod
    def validate(cls):
        if not cls.GRID_API_KEY:
            raise RuntimeError("GRID_API_KEY environment variable is required.")
