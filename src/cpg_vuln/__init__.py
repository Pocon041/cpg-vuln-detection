"""CPG-based vulnerability detection experiments."""

from cpg_vuln.platform_compat import patch_platform_uname_for_windows

patch_platform_uname_for_windows()

__version__ = "0.1.0"
