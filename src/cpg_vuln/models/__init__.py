"""Graph neural network models for the experiment matrix."""

from cpg_vuln.models.devign import DevignCPG
from cpg_vuln.models.gcn import GCNClassifier
from cpg_vuln.models.ramp_v2 import RampV2CPG, RampV2DualHeadCPG
from cpg_vuln.models.selective_fusion import SelectiveFusionCPG

__all__ = [
    "DevignCPG",
    "GCNClassifier",
    "RampV2CPG",
    "RampV2DualHeadCPG",
    "SelectiveFusionCPG",
]
