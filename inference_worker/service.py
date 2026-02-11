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
_LINUX_INSTALL_DIR = Path("/opt/grid-inference-worker")
_SYSTEMD_UNIT = f"{_SERVICE_NAME}.service"

# macOS: launchd plist
_LAUNCHD_DIR = Path.home() / "Library" / "LaunchAgents"
_LAUNCHD_LABEL = "io.aipowergrid.worker"
_LAUNCHD_PLIST = f"{_LAUNCHD_LABEL}.plist"


def _get_exec_command() -> str:
    """Get the command to run the worker without GUI.

    Uses sys.executable directly instead of pip script wrappers — the wrapper
    .exe embeds a specific python3XX.dll path that breaks when Python is
    updated or installed differently.  sys.executable always knows its own DLL.
    """
    if getattr(sys, "frozen", False):
        exe = str(Path(sys.executable).resolve())
        args = "--no-gui"
    else:
        exe = sys.executable
        args = "-m inference_worker.cli --no-gui"
    # Quote on Windows — paths like "C:\Users\John Doe\..." need it
    if sys.platform == "win32":
        return f'"{exe}" {args}'
    return f"{exe} {args}"


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
            print("  Service is installed (systemd).")
            subprocess.run(["systemctl", "status", _SERVICE_NAME, "--no-pager", "-l"])
        elif (_SYSTEMD_USER_DIR / _SYSTEMD_UNIT).exists():
            print("  Legacy user service found. Consider reinstalling as system service.")
            subprocess.run(["systemctl", "--user", "status", _SERVICE_NAME, "--no-pager", "-l"])


def schedule_start():
    """Start the service after a brief delay (allows current process to release port 7861).

    On Linux, the delayed start is baked into the install command (single pkexec prompt),
    so this is only needed for Windows and macOS.
    """
    import subprocess
    if sys.platform == "win32":
        if getattr(sys, "frozen", False):
            exe, args = str(Path(sys.executable).resolve()), "--no-gui"
        else:
            exe, args = sys.executable, "-m inference_worker.cli --no-gui"
        subprocess.Popen(
            f'cmd /c ping -n 4 127.0.0.1 >nul 2>&1 & "{exe}" {args}',
            creationflags=0x08000000 | 0x00000200,  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        )
    elif sys.platform == "darwin":
        plist = _LAUNCHD_DIR / _LAUNCHD_PLIST
        subprocess.Popen(
            ["bash", "-c", f"sleep 2 && launchctl load '{plist}'"],
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
    import getpass
    import subprocess
    import tempfile
    username = getpass.getuser()

    # For frozen binaries, copy to a stable system location so the service
    # survives even if the user moves/deletes the original download.
    frozen = getattr(sys, "frozen", False)
    if frozen:
        install_bin = _LINUX_INSTALL_DIR / "grid-inference-worker"
        exec_cmd = f"{install_bin} --no-gui"
        work_dir = str(_LINUX_INSTALL_DIR)
    else:
        exec_cmd = _get_exec_command()
        work_dir = str(Path.cwd())

    unit_content = f"""[Unit]
Description={_SERVICE_DESC}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={username}
WorkingDirectory={work_dir}
ExecStart={exec_cmd}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    unit_path = _SYSTEMD_SYSTEM_DIR / _SYSTEMD_UNIT

    # Write unit to temp file, then copy with elevated privileges
    tmp = Path(tempfile.mktemp(suffix=".service"))
    tmp.write_text(unit_content)

    cmds = ""
    if frozen:
        src_bin = str(Path(sys.executable).resolve())
        cmds += (
            f"mkdir -p '{_LINUX_INSTALL_DIR}' && "
            f"cp '{src_bin}' '{install_bin}' && "
            f"chmod 755 '{install_bin}' && "
        )
    cmds += (
        f"cp '{tmp}' '{unit_path}' && "
        f"systemctl daemon-reload && "
        f"systemctl enable {_SERVICE_NAME}"
    )
    if start:
        cmds += f" && systemctl start {_SERVICE_NAME}"
    else:
        # Delayed start — background process waits for caller to free port 7861
        cmds += (
            f" && nohup bash -c 'sleep 3 && systemctl start {_SERVICE_NAME}'"
            f" >/dev/null 2>&1 &"
        )

    try:
        if os.geteuid() == 0:
            subprocess.run(["bash", "-c", cmds], check=True, capture_output=True)
        else:
            result = subprocess.run(["pkexec", "bash", "-c", cmds], capture_output=True)
            if result.returncode != 0:
                tmp.unlink(missing_ok=True)
                if verbose:
                    print("  Authentication cancelled or pkexec failed.")
                return False
        tmp.unlink(missing_ok=True)
        if verbose:
            print(f"  System service installed.")
            print()
            print(f"  Commands:")
            print(f"    sudo systemctl status {_SERVICE_NAME}")
            print(f"    sudo systemctl stop {_SERVICE_NAME}")
            print(f"    sudo systemctl restart {_SERVICE_NAME}")
            print(f"    journalctl -u {_SERVICE_NAME} -f")
            print()
            print(f"  To remove: sudo grid-inference-worker --uninstall-service")
        return True
    except FileNotFoundError:
        tmp.unlink(missing_ok=True)
        if verbose:
            print("  pkexec not found. Install with sudo instead:")
            print(f"    sudo grid-inference-worker --install-service")
        return False
    except Exception as e:
        tmp.unlink(missing_ok=True)
        if verbose:
            print(f"  Error: {e}")
        return False


def _linux_uninstall(verbose: bool = True) -> bool:
    import subprocess

    system_unit = _SYSTEMD_SYSTEM_DIR / _SYSTEMD_UNIT
    if system_unit.exists():
        cmds = (
            f"systemctl stop {_SERVICE_NAME}; "
            f"systemctl disable {_SERVICE_NAME}; "
            f"rm -f '{system_unit}'; "
            f"rm -rf '{_LINUX_INSTALL_DIR}'; "
            f"systemctl daemon-reload"
        )
        try:
            if os.geteuid() == 0:
                subprocess.run(["bash", "-c", cmds], capture_output=True)
            else:
                result = subprocess.run(["pkexec", "bash", "-c", cmds], capture_output=True)
                if result.returncode != 0:
                    if verbose:
                        print("  Authentication cancelled or pkexec failed.")
                    return False
            if verbose:
                print("  System service stopped and removed.")
            return True
        except FileNotFoundError:
            if verbose:
                print("  pkexec not found. Remove with sudo instead:")
                print(f"    sudo grid-inference-worker --uninstall-service")
            return False
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            return False

    # Legacy: clean up old user services
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
