from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Literal

from cpg_vuln.data.graphml import GraphNode, ParsedGraph, ast_closure
from cpg_vuln.features.lexer import LexToken, TokenKind, lex_c
from cpg_vuln.features.text import normalize_node_text


NormalizationMode = Literal["raw", "semantic-anon", "full-anon"]
VALID_NORMALIZATION_MODES = {"raw", "semantic-anon", "full-anon"}

C_KEYWORDS = {
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
}

BASIC_TYPES = {
    "bool",
    "char",
    "double",
    "float",
    "int",
    "long",
    "short",
    "signed",
    "size_t",
    "ssize_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "unsigned",
    "void",
}

SEMANTIC_STEMS = {
    "src": "SEM_SRC",
    "source": "SEM_SRC",
    "from": "SEM_SRC",
    "input": "SEM_SRC",
    "in": "SEM_SRC",
    "dst": "SEM_DST",
    "dest": "SEM_DST",
    "destination": "SEM_DST",
    "to": "SEM_DST",
    "output": "SEM_DST",
    "out": "SEM_DST",
    "len": "SEM_LEN",
    "length": "SEM_LEN",
    "nbytes": "SEM_LEN",
    "bytes": "SEM_LEN",
    "size": "SEM_SIZE",
    "sz": "SEM_SIZE",
    "capacity": "SEM_SIZE",
    "cap": "SEM_SIZE",
    "count": "SEM_COUNT",
    "cnt": "SEM_COUNT",
    "num": "SEM_COUNT",
    "nb": "SEM_COUNT",
    "nr": "SEM_COUNT",
    "idx": "SEM_INDEX",
    "index": "SEM_INDEX",
    "i": "SEM_INDEX",
    "j": "SEM_INDEX",
    "k": "SEM_INDEX",
    "pos": "SEM_INDEX",
    "offset": "SEM_OFFSET",
    "off": "SEM_OFFSET",
    "start": "SEM_OFFSET",
    "end": "SEM_OFFSET",
    "buf": "SEM_BUF",
    "buffer": "SEM_BUF",
    "data": "SEM_BUF",
    "packet": "SEM_BUF",
    "payload": "SEM_BUF",
    "ptr": "SEM_PTR",
    "p": "SEM_PTR",
    "pointer": "SEM_PTR",
    "width": "SEM_DIM",
    "height": "SEM_DIM",
    "stride": "SEM_DIM",
    "pitch": "SEM_DIM",
}


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class NormalizationSpec:
    mode: str = "raw"
    version: int = 1
    api_taxonomy_version: int = 1
    tokenizer_version: int = 2

    def __post_init__(self) -> None:
        if self.mode not in VALID_NORMALIZATION_MODES:
            raise ValueError(f"unsupported normalization mode: {self.mode}")

    @property
    def normalization_key(self) -> str:
        return f"{self.mode}-v{self.version}"

    @property
    def fingerprint(self) -> str:
        return sha256_json(
            {
                "mode": self.mode,
                "version": self.version,
                "api_taxonomy_version": self.api_taxonomy_version,
                "tokenizer_version": self.tokenizer_version,
            }
        )


@dataclass(frozen=True)
class ApiInfo:
    name: str
    category: str
    keep_raw_name: bool

    def normalized_tokens(self, mode: str) -> tuple[str, ...]:
        if mode == "raw":
            return (self.name,)
        if mode == "semantic-anon" and self.keep_raw_name:
            return (self.name, self.category)
        return (self.category,)


@dataclass(frozen=True)
class ApiTaxonomy:
    entries: dict[str, ApiInfo] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "ApiTaxonomy":
        standard = {
            "memcpy": "API_COPY",
            "memmove": "API_COPY",
            "strcpy": "API_COPY",
            "strncpy": "API_COPY",
            "strcat": "API_COPY",
            "sprintf": "API_FORMAT",
            "snprintf": "API_FORMAT",
            "strlen": "API_LEN",
            "malloc": "API_ALLOC",
            "calloc": "API_ALLOC",
            "realloc": "API_REALLOC",
            "free": "API_FREE",
            "gets": "API_INPUT",
            "scanf": "API_INPUT",
            "read": "API_INPUT",
            "recv": "API_INPUT",
            "memset": "API_MEMSET",
            "fread": "API_INPUT",
            "fwrite": "API_OUTPUT",
            "open": "API_OPEN",
            "close": "API_CLOSE",
        }
        wrappers = {
            "av_malloc": "API_ALLOC",
            "av_mallocz": "API_ALLOC",
            "av_realloc": "API_REALLOC",
            "g_malloc": "API_ALLOC",
            "g_realloc": "API_REALLOC",
            "OPENSSL_malloc": "API_ALLOC",
            "OPENSSL_realloc": "API_REALLOC",
            "OPENSSL_free": "API_FREE",
        }
        entries = {
            name: ApiInfo(name=name, category=category, keep_raw_name=True)
            for name, category in standard.items()
        }
        entries.update(
            {
                name: ApiInfo(name=name, category=category, keep_raw_name=False)
                for name, category in wrappers.items()
            }
        )
        return cls(entries=entries)

    def classify(self, name: str) -> ApiInfo | None:
        return self.entries.get(name)


@dataclass(frozen=True)
class SymbolInfo:
    original_name: str
    placeholder: str
    kind: str
    tags: tuple[str, ...] = ()
    type_name: str = ""

    def tokens(self, mode: str) -> tuple[str, ...]:
        if mode == "semantic-anon":
            return (self.placeholder, *self.tags)
        return (self.placeholder,)


@dataclass(frozen=True)
class ScopeContext:
    parameters: dict[str, SymbolInfo] = field(default_factory=dict)
    locals: dict[str, SymbolInfo] = field(default_factory=dict)
    fields: dict[str, SymbolInfo] = field(default_factory=dict)
    functions: dict[str, SymbolInfo] = field(default_factory=dict)
    types: dict[str, SymbolInfo] = field(default_factory=dict)
    globals: dict[str, SymbolInfo] = field(default_factory=dict)
    unknowns: dict[str, SymbolInfo] = field(default_factory=dict)


def stable_unknown_placeholder(name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"UNKNOWN_ID_{digest}"


def stable_function_placeholder(name: str, *, semantic: bool) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    prefix = "USER_FUNC" if semantic else "FUNC"
    return f"{prefix}_{digest}"


def split_identifier(name: str) -> list[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).replace("-", "_").split("_")
    return [part.lower() for part in parts if part]


def semantic_tags(name: str) -> tuple[str, ...]:
    tags: list[str] = []
    for part in split_identifier(name):
        tag = SEMANTIC_STEMS.get(part)
        if tag is not None and tag not in tags:
            tags.append(tag)
    return tuple(tags)


def type_identifiers(type_name: str) -> list[str]:
    return [
        token.text
        for token in lex_c(type_name)
        if token.kind is TokenKind.IDENTIFIER
        and token.text not in BASIC_TYPES
        and token.text not in C_KEYWORDS
    ]


def build_scope_context(
    graph: ParsedGraph,
    root: GraphNode,
    spec: NormalizationSpec,
    *,
    api_taxonomy: ApiTaxonomy | None = None,
) -> ScopeContext:
    api_taxonomy = api_taxonomy or ApiTaxonomy.default()
    closure = ast_closure(graph, root.node_id)
    nodes = sorted(
        (node for node in graph.nodes.values() if node.node_id in closure),
        key=stable_node_key,
    )
    parameters = collect_symbols(nodes, labels={"METHOD_PARAMETER_IN"}, prefix="PARAM", kind="parameter")
    locals_ = collect_symbols(nodes, labels={"LOCAL"}, prefix="VAR", kind="local")
    fields = collect_fields(nodes)
    functions = collect_user_functions(nodes, api_taxonomy, spec)
    types = collect_types(nodes, spec)
    unknowns = collect_unknowns(nodes, parameters, locals_, fields, functions, types, api_taxonomy)
    return ScopeContext(
        parameters=parameters,
        locals=locals_,
        fields=fields,
        functions=functions,
        types=types,
        globals={},
        unknowns=unknowns,
    )


def collect_symbols(
    nodes: list[GraphNode],
    *,
    labels: set[str],
    prefix: str,
    kind: str,
) -> dict[str, SymbolInfo]:
    result: dict[str, SymbolInfo] = {}
    for node in nodes:
        if node.label not in labels:
            continue
        name = node_name(node)
        if not name or name in result:
            continue
        result[name] = SymbolInfo(
            original_name=name,
            placeholder=f"{prefix}_{len(result) + 1}",
            kind=kind,
            tags=semantic_tags(name),
            type_name=node.attrs.get("TYPE_FULL_NAME", ""),
        )
    return result


def collect_fields(nodes: list[GraphNode]) -> dict[str, SymbolInfo]:
    result: dict[str, SymbolInfo] = {}
    for node in nodes:
        if node.label != "FIELD_IDENTIFIER":
            continue
        name = (
            node.attrs.get("CANONICAL_NAME")
            or node.attrs.get("NAME")
            or node.attrs.get("CODE", "")
        ).strip()
        if not name or name in result:
            continue
        result[name] = SymbolInfo(
            original_name=name,
            placeholder=f"FIELD_{len(result) + 1}",
            kind="field",
            tags=semantic_tags(name),
        )
    return result


def collect_user_functions(
    nodes: list[GraphNode],
    api_taxonomy: ApiTaxonomy,
    spec: NormalizationSpec,
) -> dict[str, SymbolInfo]:
    result: dict[str, SymbolInfo] = {}
    for node in nodes:
        if node.label != "CALL":
            continue
        name = (node.attrs.get("NAME") or node.attrs.get("METHOD_FULL_NAME") or "").strip()
        if not name or api_taxonomy.classify(name) is not None or name in result:
            continue
        prefix = "USER_FUNC" if spec.mode == "semantic-anon" else "FUNC"
        result[name] = SymbolInfo(
            original_name=name,
            placeholder=f"{prefix}_{len(result) + 1}",
            kind="function",
        )
    return result


def collect_types(nodes: list[GraphNode], spec: NormalizationSpec) -> dict[str, SymbolInfo]:
    result: dict[str, SymbolInfo] = {}
    prefix = "USER_TYPE" if spec.mode in {"raw", "semantic-anon"} else "TYPE"
    for node in nodes:
        for name in type_identifiers(node.attrs.get("TYPE_FULL_NAME", "")):
            if name not in result:
                result[name] = SymbolInfo(
                    original_name=name,
                    placeholder=f"{prefix}_{len(result) + 1}",
                    kind="type",
                )
    return result


def collect_unknowns(
    nodes: list[GraphNode],
    parameters: dict[str, SymbolInfo],
    locals_: dict[str, SymbolInfo],
    fields: dict[str, SymbolInfo],
    functions: dict[str, SymbolInfo],
    types: dict[str, SymbolInfo],
    api_taxonomy: ApiTaxonomy,
) -> dict[str, SymbolInfo]:
    known = set(parameters) | set(locals_) | set(fields) | set(functions) | set(types)
    result: dict[str, SymbolInfo] = {}
    for node in nodes:
        code = node.attrs.get("CODE", "")
        if node.label == "CALL":
            call_name = (node.attrs.get("NAME") or node.attrs.get("METHOD_FULL_NAME") or "").strip()
            if call_name:
                known.add(call_name)
        for token in lex_c(code):
            if token.kind is not TokenKind.IDENTIFIER:
                continue
            text = token.text
            if (
                text in C_KEYWORDS
                or text in BASIC_TYPES
                or text in {"NULL", "true", "false"}
                or text in known
                or api_taxonomy.classify(text) is not None
            ):
                continue
            result.setdefault(
                text,
                SymbolInfo(
                    original_name=text,
                    placeholder=stable_unknown_placeholder(text),
                    kind="unknown",
                    tags=semantic_tags(text),
                ),
            )
    return dict(sorted(result.items(), key=lambda item: item[1].placeholder))


def node_name(node: GraphNode) -> str:
    name = node.attrs.get("NAME", "").strip()
    if name:
        return name
    parts = node.attrs.get("CODE", "").split()
    return parts[-1].strip() if parts else ""


def stable_node_key(node: GraphNode) -> tuple[int, int, tuple[int, int | str], str]:
    return (
        int_attr(node, "LINE_NUMBER"),
        int_attr(node, "COLUMN_NUMBER"),
        node_sort_key(node.node_id),
        node.attrs.get("NAME", node.attrs.get("CODE", "")),
    )


def int_attr(node: GraphNode, key: str) -> int:
    try:
        return int(node.attrs.get(key, "1000000000"))
    except ValueError:
        return 1000000000


def node_sort_key(node_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(node_id))
    except ValueError:
        return (1, node_id)


class IdentifierSemanticNormalizer:
    def __init__(
        self,
        spec: NormalizationSpec | None = None,
        *,
        api_taxonomy: ApiTaxonomy | None = None,
    ) -> None:
        self.spec = spec or NormalizationSpec(mode="raw")
        self.api_taxonomy = api_taxonomy or ApiTaxonomy.default()

    def normalize_node(self, node: GraphNode, scope: ScopeContext | None = None) -> str:
        if self.spec.mode == "raw":
            return normalize_node_text(node)
        if scope is None:
            raise ValueError(f"{self.spec.mode} normalization requires a frozen ScopeContext")
        if node.label == "BLOCK":
            return "<BLOCK>"
        if node.label == "METHOD":
            return self.normalize_method(node, scope)
        if node.label == "METHOD_RETURN":
            return self.normalize_return_type(node, scope)
        code = node.attrs.get("CODE", "").strip() or node.attrs.get("NAME", "").strip()
        if not code:
            return f"<{node.label}>"
        return self.normalize_code(
            code,
            scope,
            call_name=node.attrs.get("NAME") if node.label == "CALL" else None,
        )

    def normalize_code(
        self,
        code: str,
        scope: ScopeContext,
        *,
        call_name: str | None = None,
    ) -> str:
        output: list[str] = []
        for index, token in enumerate(lex_c(code)):
            output.extend(self.normalize_token(token, scope, call_name=call_name, token_index=index))
        return " ".join(item for item in output if item)

    def normalize_token(
        self,
        token: LexToken,
        scope: ScopeContext,
        *,
        call_name: str | None,
        token_index: int,
    ) -> tuple[str, ...]:
        text = token.text
        if token.kind is TokenKind.STRING_LITERAL:
            return ("STR",)
        if token.kind is TokenKind.CHAR_LITERAL:
            return ("CHAR",)
        if token.kind is TokenKind.HEX_LITERAL:
            return ("HEX",)
        if token.kind is TokenKind.FLOAT_LITERAL:
            return ("FLOAT",)
        if token.kind is TokenKind.INTEGER_LITERAL:
            return (self.normalize_integer(text),)
        if token.kind is TokenKind.JOERN_OPERATOR:
            return (text,)
        if token.kind is not TokenKind.IDENTIFIER:
            return (text,)
        return self.normalize_identifier(text, scope, call_name=call_name, token_index=token_index)

    def normalize_identifier(
        self,
        text: str,
        scope: ScopeContext,
        *,
        call_name: str | None,
        token_index: int,
    ) -> tuple[str, ...]:
        if text in C_KEYWORDS or text in BASIC_TYPES or text in {"NULL", "true", "false"}:
            return (text,)
        parameter = scope.parameters.get(text)
        if parameter is not None:
            return parameter.tokens(self.spec.mode)
        local = scope.locals.get(text)
        if local is not None:
            return local.tokens(self.spec.mode)
        field = scope.fields.get(text)
        if field is not None:
            return field.tokens(self.spec.mode)
        type_info = scope.types.get(text)
        if type_info is not None:
            return type_info.tokens(self.spec.mode)
        api = self.api_taxonomy.classify(text)
        if api is not None:
            return api.normalized_tokens(self.spec.mode)
        function = scope.functions.get(text)
        if function is not None:
            return function.tokens(self.spec.mode)
        unknown = scope.unknowns.get(text)
        if unknown is not None:
            return unknown.tokens(self.spec.mode)
        if call_name == text and token_index == 0:
            return (stable_function_placeholder(text, semantic=self.spec.mode == "semantic-anon"),)
        return (stable_unknown_placeholder(text),)

    def normalize_integer(self, text: str) -> str:
        stripped = re.sub(r"[uUlL]+$", "", text)
        if stripped in {"0", "1", "2"}:
            return f"NUM_{stripped}"
        return "NUM"

    def normalize_method(self, node: GraphNode, scope: ScopeContext) -> str:
        signature = node.attrs.get("SIGNATURE", "")
        if not signature:
            return "<METHOD>"
        normalized = self.normalize_code(signature, scope)
        return f"<METHOD {normalized}>"

    def normalize_return_type(self, node: GraphNode, scope: ScopeContext) -> str:
        type_name = node.attrs.get("TYPE_FULL_NAME", "").strip()
        if not type_name:
            return "<RET>"
        normalized = self.normalize_code(type_name, scope)
        return f"<RET {normalized}>"


def normalize_source_text(
    source: str,
    scope: ScopeContext,
    spec: NormalizationSpec,
    *,
    api_taxonomy: ApiTaxonomy | None = None,
) -> str:
    if spec.mode == "raw":
        return source
    return IdentifierSemanticNormalizer(
        spec,
        api_taxonomy=api_taxonomy,
    ).normalize_code(source, scope)
