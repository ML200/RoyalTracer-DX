"""Parser for PBRT v4 scene files.

Produces a flat list of Statements that the scene builder then executes
in order, maintaining a CTM and attribute stack.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pbrt_lex import tokenize, to_num, LexError


@dataclass
class Statement:
    name: str
    args: List[Any] = field(default_factory=list)
    params: Dict[str, "Param"] = field(default_factory=dict)
    line: int = 0


@dataclass
class Param:
    type: str   # e.g. 'rgb', 'float', 'integer', 'string', 'texture', 'point3', ...
    name: str   # parameter name, e.g. 'reflectance'
    values: List[Any]  # list of numbers or strings


class ParseError(Exception):
    pass


# Directives that take no arguments
_NULLARY = {
    "AttributeBegin", "AttributeEnd",
    "WorldBegin",
    "ObjectEnd",
    "ReverseOrientation",
    "Identity",
}

# Directives whose body is: type-string, then parameter list
_TYPED_PARAMS = {
    "Camera", "Sampler", "Film", "Filter", "PixelFilter", "Integrator",
    "Material", "Shape", "LightSource", "AreaLightSource",
    "Accelerator", "MakeNamedMedium", "ColorSpace", "Option",
}

# Directives whose body is: name string
_NAME_ONLY = {
    "NamedMaterial", "ObjectInstance", "Include", "Import",
    "CoordinateSystem", "CoordSysTransform",
}


class Parser:
    def __init__(self, tokens, source="<input>"):
        self.toks = tokens
        self.i = 0
        self.source = source

    # ---- token helpers ----------------------------------------------------

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _advance(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def _err(self, msg, line=0):
        raise ParseError(f"{self.source}:{line}: {msg}")

    def _expect(self, kind):
        t = self._peek()
        if t is None or t[0] != kind:
            got = "EOF" if t is None else t[0]
            self._err(f"expected {kind}, got {got}", t[2] if t else 0)
        return self._advance()

    # ---- value reading ----------------------------------------------------

    def _read_num(self):
        t = self._expect("NUMBER")
        return to_num(t[1])

    def _read_str(self):
        t = self._expect("STRING")
        return t[1]

    def _read_list(self):
        self._expect("LBRACK")
        vals = []
        while True:
            t = self._peek()
            if t is None:
                self._err("unterminated list", 0)
            if t[0] == "RBRACK":
                break
            if t[0] == "NUMBER":
                vals.append(to_num(t[1]))
                self._advance()
            elif t[0] == "STRING":
                vals.append(t[1])
                self._advance()
            elif t[0] == "IDENT" and t[1] in ("true", "false"):
                vals.append(t[1] == "true")
                self._advance()
            else:
                self._err(f"unexpected {t[0]} {t[1]!r} in list", t[2])
        self._expect("RBRACK")
        return vals

    def _read_value(self):
        """Read a parameter value: a single literal or a bracketed list."""
        t = self._peek()
        if t is None:
            self._err("expected value, got EOF", 0)
        if t[0] == "LBRACK":
            return self._read_list()
        if t[0] == "NUMBER":
            self._advance()
            return [to_num(t[1])]
        if t[0] == "STRING":
            self._advance()
            return [t[1]]
        if t[0] == "IDENT" and t[1] in ("true", "false"):
            self._advance()
            return [t[1] == "true"]
        self._err(f"expected value, got {t[0]} {t[1]!r}", t[2])

    def _read_floats(self, n):
        return [self._read_num() for _ in range(n)]

    def _read_params(self):
        params = {}
        while True:
            t = self._peek()
            if t is None or t[0] != "STRING":
                break
            self._advance()
            decl = t[1].strip()
            parts = decl.split(None, 1)
            if len(parts) == 2:
                ptype, pname = parts[0], parts[1].strip()
            else:
                # PBRT v4 allows omitting the type for some well-known names,
                # but in practice every official scene declares both. Treat as
                # un-typed and let downstream code infer.
                ptype, pname = "", parts[0]
            value = self._read_value()
            params[pname] = Param(type=ptype, name=pname, values=value)
        return params

    # ---- statement parsing ------------------------------------------------

    def parse(self):
        stmts = []
        while self.i < len(self.toks):
            t = self._peek()
            if t[0] != "IDENT":
                self._err(f"expected directive, got {t[0]} {t[1]!r}", t[2])
            stmts.append(self._parse_stmt())
        return stmts

    def _parse_stmt(self):
        t = self._advance()
        name = t[1]
        line = t[2]

        if name in _NULLARY:
            return Statement(name, line=line)

        if name == "Translate":
            return Statement(name, args=self._read_floats(3), line=line)
        if name == "Scale":
            return Statement(name, args=self._read_floats(3), line=line)
        if name == "Rotate":
            return Statement(name, args=self._read_floats(4), line=line)
        if name == "LookAt":
            return Statement(name, args=self._read_floats(9), line=line)
        if name in ("Transform", "ConcatTransform"):
            if self._peek() and self._peek()[0] == "LBRACK":
                vals = self._read_list()
            else:
                vals = self._read_floats(16)
            if len(vals) != 16:
                self._err(f"{name} requires 16 numbers, got {len(vals)}", line)
            return Statement(name, args=vals, line=line)

        if name in _NAME_ONLY:
            return Statement(name, args=[self._read_str()], line=line)

        if name == "ObjectBegin":
            return Statement(name, args=[self._read_str()], line=line)

        if name == "MakeNamedMaterial":
            mat_name = self._read_str()
            params = self._read_params()
            return Statement(name, args=[mat_name], params=params, line=line)

        if name == "Texture":
            tname = self._read_str()
            ttype = self._read_str()
            tclass = self._read_str()
            params = self._read_params()
            return Statement(name, args=[tname, ttype, tclass], params=params, line=line)

        if name == "MediumInterface":
            inner = self._read_str()
            outer = self._read_str()
            return Statement(name, args=[inner, outer], line=line)

        if name == "ActiveTransform":
            t2 = self._advance()
            return Statement(name, args=[t2[1]], line=line)

        if name == "TransformTimes":
            return Statement(name, args=self._read_floats(2), line=line)

        if name == "Sides":
            # PBRT v4 'Sides "single"|"both"' - take next ident or string
            t2 = self._advance()
            return Statement(name, args=[t2[1]], line=line)

        if name in _TYPED_PARAMS:
            type_name = self._read_str()
            params = self._read_params()
            return Statement(name, args=[type_name], params=params, line=line)

        # Unknown directive: skip until next IDENT to keep parsing useful
        skipped = []
        while self.i < len(self.toks):
            t2 = self._peek()
            if t2[0] == "IDENT":
                break
            skipped.append(self._advance())
        return Statement(name, args=[("__skipped__", skipped)], line=line)


def parse_text(text, source="<input>"):
    toks = tokenize(text, source=source)
    return Parser(toks, source=source).parse()


def parse_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return parse_text(text, source=str(path))
