from __future__ import annotations

import os
import platform
import socket
import sys


def patch_platform_uname_for_windows() -> None:
    """Avoid slow or hung WMI lookups when libraries call platform.machine()."""
    if sys.platform != "win32":
        return
    if getattr(platform, "_uname_cache", None) is not None:
        return
    machine = (
        os.environ.get("PROCESSOR_ARCHITEW6432")
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or "AMD64"
    )
    node = os.environ.get("COMPUTERNAME") or _hostname()
    release, version = _windows_version()
    platform._uname_cache = platform.uname_result(
        "Windows",
        node,
        release,
        version,
        machine,
    )


def _hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return ""


def _windows_version() -> tuple[str, str]:
    try:
        version = sys.getwindowsversion()
    except AttributeError:
        return "", ""
    release = str(version.major)
    full_version = f"{version.major}.{version.minor}.{version.build}"
    return release, full_version
