from __future__ import annotations

import platform
import sys

from cpg_vuln.platform_compat import patch_platform_uname_for_windows


def test_windows_uname_patch_uses_environment_machine(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform, "_uname_cache", None)
    monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")
    monkeypatch.delenv("PROCESSOR_ARCHITEW6432", raising=False)

    patch_platform_uname_for_windows()

    assert platform.machine() == "AMD64"
