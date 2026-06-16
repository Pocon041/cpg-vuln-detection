from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


ALLOC_APIS = {"malloc", "calloc", "realloc", "av_malloc", "av_mallocz", "av_realloc"}
COPY_APIS = {"memcpy", "memmove", "strcpy", "strncpy", "strcat", "sprintf"}
RELEASE_APIS = {"free", "av_free", "av_freep", "close"}
IO_APIS = {"read", "write", "recv", "send", "scanf"}
ERROR_RETURN_TOKENS = ("error", "invalid", "einval", "averror", "return -")

MOTIF_FEATURE_NAMES = [
    "known_alloc_api",
    "known_copy_api",
    "known_release_api",
    "known_io_api",
    "indirect_call",
    "array_index",
    "field_access",
    "address_of",
    "integer_mul",
    "integer_sub",
    "division",
    "modulo",
    "bit_shift",
    "num_nodes",
    "num_edges",
    "num_ast_edges",
    "num_cfg_edges",
    "num_cdg_edges",
    "num_reaching_def_edges",
    "branch_count",
    "loop_count",
    "return_count",
    "error_return_count",
]

MOTIF_FEATURE_GROUPS = {
    "motif": [
        "known_alloc_api",
        "known_copy_api",
        "known_release_api",
        "known_io_api",
        "indirect_call",
        "array_index",
        "field_access",
        "address_of",
        "integer_mul",
        "integer_sub",
        "division",
        "modulo",
        "bit_shift",
    ],
    "structure": [
        "num_ast_edges",
        "num_cfg_edges",
        "num_cdg_edges",
        "num_reaching_def_edges",
    ],
    "scale": [
        "num_nodes",
        "num_edges",
        "branch_count",
        "loop_count",
        "return_count",
        "error_return_count",
    ],
}


def motif_feature_names() -> list[str]:
    return list(MOTIF_FEATURE_NAMES)


def motif_feature_groups() -> dict[str, list[str]]:
    return {name: list(values) for name, values in MOTIF_FEATURE_GROUPS.items()}


@dataclass(frozen=True)
class RiskMotif:
    sample_id: str
    label: int
    features: dict[str, float]

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            [self.features.get(name, 0.0) for name in motif_feature_names()],
            dtype=np.float32,
        )


def extract_risk_motif(payload: dict[str, object]) -> RiskMotif:
    features = {name: 0.0 for name in motif_feature_names()}
    node_labels = [str(value) for value in payload.get("node_labels", [])]
    node_names = [str(value) for value in payload.get("node_names", [])]
    method_full_names = [str(value) for value in payload.get("method_full_names", [])]
    code_summaries = [str(value).lower() for value in payload.get("code_summaries", [])]
    control_types = [str(value).upper() for value in payload.get("control_structure_types", [])]

    call_names: set[str] = set()
    for label, name, method_full_name in zip(
        node_labels,
        node_names,
        method_full_names,
        strict=False,
    ):
        if label != "CALL":
            continue
        if name:
            call_names.add(name)
        if method_full_name:
            call_names.add(method_full_name)

    api_names = {name.rsplit(".", 1)[-1].lower() for name in call_names}
    features["known_alloc_api"] = float(bool(api_names & ALLOC_APIS))
    features["known_copy_api"] = float(bool(api_names & COPY_APIS))
    features["known_release_api"] = float(bool(api_names & RELEASE_APIS))
    features["known_io_api"] = float(bool(api_names & IO_APIS))
    features["indirect_call"] = float("<operator>.pointerCall" in call_names)
    features["array_index"] = float("<operator>.indexAccess" in call_names)
    features["field_access"] = float(
        "<operator>.fieldAccess" in call_names
        or "<operator>.indirectFieldAccess" in call_names
    )
    features["address_of"] = float("<operator>.addressOf" in call_names)
    features["integer_mul"] = float("<operator>.multiplication" in call_names)
    features["integer_sub"] = float("<operator>.subtraction" in call_names)
    features["division"] = float("<operator>.division" in call_names)
    features["modulo"] = float("<operator>.modulo" in call_names)
    features["bit_shift"] = float(
        "<operator>.shiftLeft" in call_names
        or "<operator>.arithmeticShiftRight" in call_names
        or "<operator>.logicalShiftRight" in call_names
    )

    node_type_histogram = payload.get("node_type_histogram", {})
    features["branch_count"] = math.log1p(
        int(node_type_histogram.get("CONTROL_STRUCTURE", 0))
        if isinstance(node_type_histogram, dict)
        else 0
    )
    features["loop_count"] = math.log1p(
        sum(kind in {"FOR", "WHILE", "DO"} for kind in control_types)
    )
    features["return_count"] = math.log1p(
        int(node_type_histogram.get("RETURN", 0))
        if isinstance(node_type_histogram, dict)
        else 0
    )
    return_codes = [
        code
        for label, code in zip(node_labels, code_summaries, strict=False)
        if label == "RETURN"
    ]
    features["error_return_count"] = math.log1p(
        sum(code.count(token) for code in return_codes for token in ERROR_RETURN_TOKENS)
    )

    node_type_id = payload.get("node_type_id")
    edge_index = payload.get("edge_index")
    features["num_nodes"] = math.log1p(
        int(node_type_id.numel())
        if isinstance(node_type_id, torch.Tensor)
        else len(node_labels)
    )
    features["num_edges"] = math.log1p(
        int(edge_index.shape[1]) if isinstance(edge_index, torch.Tensor) else 0
    )

    edge_type_histogram = payload.get("edge_type_histogram", {})
    edge_type = payload.get("edge_type")
    edge_type_names = payload.get("edge_type_names", {})
    if isinstance(edge_type_histogram, dict):
        features["num_ast_edges"] = math.log1p(int(edge_type_histogram.get("AST", 0)))
        features["num_cfg_edges"] = math.log1p(int(edge_type_histogram.get("CFG", 0)))
        features["num_cdg_edges"] = math.log1p(int(edge_type_histogram.get("CDG", 0)))
        features["num_reaching_def_edges"] = math.log1p(
            int(edge_type_histogram.get("REACHING_DEF", 0))
        )
    elif isinstance(edge_type, torch.Tensor) and isinstance(edge_type_names, dict):
        for relation, key in (
            ("AST", "num_ast_edges"),
            ("CFG", "num_cfg_edges"),
            ("CDG", "num_cdg_edges"),
            ("REACHING_DEF", "num_reaching_def_edges"),
        ):
            relation_id = edge_type_names.get(relation)
            features[key] = (
                math.log1p(int((edge_type == relation_id).sum()))
                if relation_id is not None
                else 0.0
            )

    label_tensor = payload.get("y")
    label = (
        int(label_tensor.view(-1)[0])
        if isinstance(label_tensor, torch.Tensor)
        else int(payload.get("label", 0))
    )
    return RiskMotif(sample_id=str(payload["sample_id"]), label=label, features=features)
