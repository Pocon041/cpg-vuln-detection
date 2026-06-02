"""Graph neural network models for the experiment matrix."""

from cpg_vuln.models.gcn import GCNClassifier
from cpg_vuln.models.selective_fusion import SelectiveFusionCPG

__all__ = ["GCNClassifier", "SelectiveFusionCPG"]

