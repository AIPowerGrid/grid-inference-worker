"""Cross-platform service installation (systemd / launchd / Windows startup)."""

import os
import sys
from pathlib import Path

_SERVICE_NAME = "grid-inference-worker"
_SERVICE_DESC = "Grid Inference Worker — AI Power Grid"

# Windows: HKCU Run key
_WIN_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_WIN_APP_NAME = "GridInferenceWorker"

# Linux: systemd unit
_SYSTEMD_USER_DIR = Path.home() / ".config" / "systemd" / "user"
_SYSTEMD_SYSTEM_DIR = Path("/etc/systemd/system")
_SYSTEMD_UNIT = f"{_SERVICE_NAME}.service"

# macOS: launchd plist
_LAUNCHD_DIR = Path.home() / "Library" / "LaunchAgents"
_LAUNCHD_LABEL = "io.aipowergrid.worker"
_LAUNCHD_PLIST = f"{_LAUNCHD_LABEL}.plist"


def _get_exec_command() -> str:
    """Get the command to run the worker without GUI."""
    if getattr(sys, "frozen", False):
        return f"{sys.executable} --no-gui"
    import shutil
    script = shutil.which("grid-inference-worker")
    if script:
        return f"{script} --no-gui"
    return f"{sys.executable} -m inference_worker.cli --no-gui"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_installed() -> bool:
    if sys.platform == "win32":
        return _win_is_installed()
    elif sys.platform == "darwin":
        return (_LAUNCHD_DIR / _LAUNCHD_PLIST).exists()
    else:
        return (
            (_SYSTEMD_USER_DIR / _SYSTEMD_UNIT).exists()
            or (_SYSTEMD_SYSTEM_DIR / _SYSTEMD_UNIT).exists()
        )


def install(verbose: bool = True, start: bool = True) -> bool:
    if sys.platform == "win32":
        return _win_install(verbose)
    elif sys.platform == "darwin":
        return _macos_install(verbose, start)
    else:
        return _linux_install(verbose, start)


def uninstall(verbose: bool = True) -> bool:
    if sys.platform == "win32":
        return _win_uninstall(verbose)
    elif sys.platform == "darwin":
        return _macos_uninstall(verbose)
    else:
        return _linux_uninstall(verbose)


def status():
    """Print service status and exit."""
    if not is_installed():
        print("  Service is not installed.")
        print("  Install with: grid-inference-worker --install-service")
        return

    if sys.platform == "win32":
        print("  Service is installed (Windows startup).")
    elif sys.platform == "darwin":
        import subprocess
        print("  Service is installed (launchd).")
        result = subprocess.run(
            ["launchctl", "list", _LAUNCHD_LABEL],
            capture_output=True, text=True,
        )
        print("  Status: running" if result.returncode == 0 else "  Status: not running")
    else:
        import subprocess
        if (_SYSTEMD_SYSTEM_DIR / _SYSTEMD_UNIT).exists():
            print("  Service is installed (systemd system).")
            subprocess.run(["systemctl", "status", _SERVICE_NAME, "--no-pager", "-l"])
        elif (_SYSTEMD_USER_DIR / _SYSTEMD_UNIT).exists():
            print("  Service is installed (systemd user).")
            subprocess.run(["systemctl", "--user", "status", _SERVICE_NAME, "--no-pager", "-l"])


def schedule_start():
    """Start the service after a brief delay (allows current process to release port 7861)."""
    import subprocess
    if sys.platform == "win32":
        if getattr(sys, "frozen", False):
            exe, args = sys.executable, "--no-gui"
        else:
            import shutil
            script = shutil.which("grid-inference-worker")
            if script:
                exe, args = script, "--no-gui"
            else:
                exe, args = sys.executable, "-m inference_worker.cli --no-gui"
        subprocess.Popen(
            f'cmd /c ping -n 4 127.0.0.1 >nul 2>&1 & "{exe}" {args}',
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
    elif sys.platform == "darwin":
        plist = _LAUNCHD_DIR / _LAUNCHD_PLIST
        subprocess.Popen(
            ["bash", "-c", f"sleep 2 && launchctl load '{plist}'"],
            start_new_session=True,
        )
    else:
        subprocess.Popen(
            ["bash", "-c", f"sleep 2 && systemctl --user start {_SERVICE_NAME}"],
            start_new_session=True,
        )


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def _win_is_installed() -> bool:
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _WIN_RUN_KEY, 0, winreg.KEY_READ)
        try:
            winreg.QueryValueEx(key, _WIN_APP_NAME)
            return True
        except OSError:
            return False
        finally:
            winreg.CloseKey(key)
    except Exception:
        return False


def _win_install(verbose: bool = True) -> bool:
    try:
        import winreg
        exe_cmd = _get_exec_command()
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _WIN_RUN_KEY, 0, winreg.KEY_SET_VALUE)
        try:
            winreg.SetValueEx(key, _WIN_APP_NAME, 0, winreg.REG_SZ, exe_cmd)
        finally:
            winreg.CloseKey(key)
        if verbose:
            print("  Service installed (Windows startup).")
            print()
            print("  The worker will start when you log in.")
            print("  To remove: grid-inference-worker --uninstall-service")
        return True
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return False


def _win_uninstall(verbose: bool = True) -> bool:
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _WIN_RUN_KEY, 0, winreg.KEY_SET_VALUE)
        try:
            winreg.DeleteValue(key, _WIN_APP_NAME)
        finally:
            winreg.CloseKey(key)
        if verbose:
            print("  Service removed from Windows startup.")
        return True
    except OSError:
        if verbose:
            print("  Service was not installed.")
        return False
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return False


# ---------------------------------------------------------------------------
# Linux (systemd)
# ---------------------------------------------------------------------------

def _linux_install(verbose: bool = True, start: bool = True) -> bool:
    import subprocess
    exec_cmd = _get_exec_command()
    work_dir = str(Path(sys.executable).resolve().parent) if getattr(sys, "frozen", False) else str(Path.cwd())

    unit_content = f"""[Unit]
Description={_SERVICE_DESC}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={work_dir}
ExecStart={exec_cmd}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
"""
    if os.geteuid() == 0:
        unit_path = _SYSTEMD_SYSTEM_DIR / _SYSTEMD_UNIT
        unit_content = unit_content.replace("WantedBy=default.target", "WantedBy=multi-user.target")
        try:
            unit_path.write_text(unit_content)
            subprocess.run(["systemctl", "daemon-reload"], check=True, capture_output=True)
            subprocess.run(["systemctl", "enable", _SERVICE_NAME], check=True, capture_output=True)
            if start:
                subprocess.run(["systemctl", "start", _SERVICE_NAME], check=True, capture_output=True)
            if verbose:
                print(f"  System service installed and started.")
                print()
                print(f"  Commands:")
                print(f"    sudo systemctl status {_SERVICE_NAME}")
                print(f"    sudo systemctl stop {_SERVICE_NAME}")
                print(f"    sudo systemctl restart {_SERVICE_NAME}")
                print(f"    journalctl -u {_SERVICE_NAME} -f")
                print()
                print(f"  To remove: sudo grid-inference-worker --uninstall-service")
            return True
        except Exception as e:
            if verbose:
                print(f"  Error installing system service: {e}")
            return False
    else:
        _SYSTEMD_USER_DIR.mkdir(parents=True, exist_ok=True)
        unit_path = _SYSTEMD_USER_DIR / _SYSTEMD_UNIT
        try:
            unit_path.write_text(unit_content)
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True)
            subprocess.run(["systemctl", "--user", "enable", _SERVICE_NAME], check=True, capture_output=True)
            if start:
                subprocess.run(["systemctl", "--user", "start", _SERVICE_NAME], check=True, capture_output=True)
            if verbose:
                print(f"  User service installed and started.")
                print()
                print(f"  Commands:")
                print(f"    systemctl --user status {_SERVICE_NAME}")
                print(f"    systemctl --user stop {_SERVICE_NAME}")
                print(f"    systemctl --user restart {_SERVICE_NAME}")
                print(f"    journalctl --user -u {_SERVICE_NAME} -f")
                print()
                print(f"  To remove: grid-inference-worker --uninstall-service")
            return True
        except Exception as e:
            if verbose:
                print(f"  Error installing user service: {e}")
            return False


def _linux_uninstall(verbose: bool = True) -> bool:
    import subprocess

    system_unit = _SYSTEMD_SYSTEM_DIR / _SYSTEMD_UNIT
    if system_unit.exists():
        try:
            subprocess.run(["systemctl", "stop", _SERVICE_NAME], capture_output=True)
            subprocess.run(["systemctl", "disable", _SERVICE_NAME], capture_output=True)
            system_unit.unlink()
            subprocess.run(["systemctl", "daemon-reload"], capture_output=True)
            if verbose:
                print("  System service stopped and removed.")
            return True
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            return False

    user_unit = _SYSTEMD_USER_DIR / _SYSTEMD_UNIT
    if user_unit.exists():
        try:
            subprocess.run(["systemctl", "--user", "stop", _SERVICE_NAME], capture_output=True)
            subprocess.run(["systemctl", "--user", "disable", _SERVICE_NAME], capture_output=True)
            user_unit.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
            if verbose:
                print("  User service stopped and removed.")
            return True
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            return False

    if verbose:
        print("  Service was not installed.")
    return False


# ---------------------------------------------------------------------------
# macOS (launchd)
# ---------------------------------------------------------------------------

def _macos_install(verbose: bool = True, start: bool = True) -> bool:
    import subprocess
    exec_parts = _get_exec_command().split()
    work_dir = str(Path(sys.executable).resolve().parent) if getattr(sys, "frozen", False) else str(Path.cwd())

    arg_entries = "\n".join(f"      <string>{a}</string>" for a in exec_parts)

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
{arg_entries}
    </array>
    <key>WorkingDirectory</key>
    <string>{work_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/{_SERVICE_NAME}.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/{_SERVICE_NAME}.err</string>
</dict>
</plist>
"""
    _LAUNCHD_DIR.mkdir(parents=True, exist_ok=True)
    plist_path = _LAUNCHD_DIR / _LAUNCHD_PLIST
    try:
        plist_path.write_text(plist_content)
        if start:
            subprocess.run(["launchctl", "load", str(plist_path)], check=True, capture_output=True)
        if verbose:
            print("  Service installed and started (launchd).")
            print()
            print("  Commands:")
            print(f"    launchctl list | grep {_LAUNCHD_LABEL}")
            print(f"    launchctl stop {_LAUNCHD_LABEL}")
            print(f"    tail -f /tmp/{_SERVICE_NAME}.log")
            print()
            print("  To remove: grid-inference-worker --uninstall-service")
        return True
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return False


def _macos_uninstall(verbose: bool = True) -> bool:
    import subprocess
    plist_path = _LAUNCHD_DIR / _LAUNCHD_PLIST
    if not plist_path.exists():
        if verbose:
            print("  Service was not installed.")
        return False
    try:
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
        plist_path.unlink()
        if verbose:
            print("  Service stopped and removed.")
        return True
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return False
