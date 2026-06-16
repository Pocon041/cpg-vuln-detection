from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cpg_vuln.platform_compat import patch_platform_uname_for_windows

patch_platform_uname_for_windows()
