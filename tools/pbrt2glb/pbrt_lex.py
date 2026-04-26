"""Tokenizer for PBRT v4 scene files."""

import re

_TOKEN_RE = re.compile(
    r"""
      (?P<COMMENT>\#[^\n]*)
    | (?P<WS>\s+)
    | (?P<STRING>"[^"]*")
    | (?P<LBRACK>\[)
    | (?P<RBRACK>\])
    | (?P<NUMBER>[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
    | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<OTHER>.)
    """,
    re.VERBOSE,
)


class LexError(Exception):
    pass


def tokenize(text, source="<input>"):
    """Return a list of (kind, value, line) tuples.

    kind in {STRING, NUMBER, IDENT, LBRACK, RBRACK}.
    STRING values are returned without surrounding quotes.
    NUMBER values are returned as the original string (parsed lazily).
    """
    tokens = []
    line = 1
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        val = m.group()
        if kind == "WS":
            line += val.count("\n")
            continue
        if kind == "COMMENT":
            continue
        if kind == "STRING":
            tokens.append(("STRING", val[1:-1], line))
            line += val.count("\n")
            continue
        if kind == "NUMBER":
            tokens.append(("NUMBER", val, line))
            continue
        if kind in ("LBRACK", "RBRACK", "IDENT"):
            tokens.append((kind, val, line))
            continue
        raise LexError(f"{source}:{line}: unexpected character {val!r}")
    return tokens


def to_num(s):
    if any(c in s for c in ".eE"):
        return float(s)
    try:
        return int(s)
    except ValueError:
        return float(s)
