from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TokenKind(str, Enum):
    JOERN_OPERATOR = "JOERN_OPERATOR"
    STRING_LITERAL = "STRING_LITERAL"
    CHAR_LITERAL = "CHAR_LITERAL"
    HEX_LITERAL = "HEX_LITERAL"
    FLOAT_LITERAL = "FLOAT_LITERAL"
    INTEGER_LITERAL = "INTEGER_LITERAL"
    IDENTIFIER = "IDENTIFIER"
    OPERATOR = "OPERATOR"


@dataclass(frozen=True)
class LexToken:
    kind: TokenKind
    text: str


_TOKEN_PATTERN = re.compile(
    r"""
    (?P<STRING>"(?:\\.|[^"\\])*")
  | (?P<CHAR>'(?:\\.|[^'\\])')
  | (?P<COMMENT>//[^\n]*|/\*.*?\*/)
  | (?P<JOERN><operator>\.[A-Za-z_][A-Za-z_0-9]*)
  | (?P<HEX>0[xX][0-9A-Fa-f]+(?:[uUlL]+)?)
  | (?P<FLOAT>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)[fFlL]?|\d+\.\d*[fFlL]?)
  | (?P<INT>\d+(?:[uUlL]+)?)
  | (?P<ID>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<OP>==|!=|<=|>=|->|\+\+|--|&&|\|\||<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|::|[{}()\[\];,.?:~!%^&*+\-=/<>|])
    """,
    re.VERBOSE | re.DOTALL,
)


def lex_c(text: str) -> list[LexToken]:
    tokens: list[LexToken] = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        match = _TOKEN_PATTERN.match(text, position)
        if match is None:
            tokens.append(LexToken(TokenKind.OPERATOR, text[position]))
            position += 1
            continue
        kind = match.lastgroup
        value = match.group()
        position = match.end()
        if kind == "COMMENT":
            continue
        if kind == "STRING":
            tokens.append(LexToken(TokenKind.STRING_LITERAL, value))
        elif kind == "CHAR":
            tokens.append(LexToken(TokenKind.CHAR_LITERAL, value))
        elif kind == "JOERN":
            tokens.append(LexToken(TokenKind.JOERN_OPERATOR, value))
        elif kind == "HEX":
            tokens.append(LexToken(TokenKind.HEX_LITERAL, value))
        elif kind == "FLOAT":
            tokens.append(LexToken(TokenKind.FLOAT_LITERAL, value))
        elif kind == "INT":
            tokens.append(LexToken(TokenKind.INTEGER_LITERAL, value))
        elif kind == "ID":
            tokens.append(LexToken(TokenKind.IDENTIFIER, value))
        else:
            tokens.append(LexToken(TokenKind.OPERATOR, value))
    return tokens
