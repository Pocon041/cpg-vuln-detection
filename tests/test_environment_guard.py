from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PREFIX = Path(r"D:\Anaconda")
EIT_PREFIX = Path(sys.prefix)


def test_environment_guard_rejects_spoofed_eit_variable_with_base_python() -> None:
    result = _run_guard(BASE_PREFIX)

    assert result.returncode != 0
    assert "Actual Python interpreter is not from EIT" in (result.stdout + result.stderr)


def test_environment_guard_accepts_real_eit_python() -> None:
    result = _run_guard(EIT_PREFIX)

    assert result.returncode == 0, result.stdout + result.stderr


def _run_guard(python_prefix: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["CONDA_DEFAULT_ENV"] = "EIT"
    environment["CONDA_PREFIX"] = str(python_prefix)
    environment["PATH"] = os.pathsep.join(
        [str(python_prefix), str(python_prefix / "Scripts"), environment["PATH"]]
    )
    command = (
        f". '{ROOT / 'scripts' / 'assert_eit.ps1'}'; "
        "Assert-EitEnvironment"
    )
    return subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
