"""Schema normalization and inference for strict CSV files."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SUPPORTED = ("int64", "int32", "float64", "float32", "string", "skip")
NUMERIC = ("int64", "int32", "float64", "float32")
SchemaSpec = Sequence[tuple[str, Any]] | Mapping[str, Any]

_ALIASES = {
    "int": "int64", "i8": "int64", "long": "int64",
    "i4": "int32",
    "float": "float64", "double": "float64", "f8": "float64",
    "f4": "float32", "single": "float32", "half": "float32",
    "str": "string", "text": "string", "object": "string", "utf8": "string",
    "none": "skip", "drop": "skip",
}

_NUMPY_TO_NAME = {
    "int64": "int64", "int32": "int32", "float64": "float64", "float32": "float32",
    # Narrow types parse into the closest supported width.
    "int8": "int32", "int16": "int32", "uint8": "int32", "uint16": "int32", "uint32": "int64",
    "float16": "float32", "bool": "int32",
}


def dtype_name(spec: Any) -> str:
    """Map a dtype spec (string, numpy dtype, torch dtype, Python type) to a supported name."""
    if spec is None:
        return "skip"
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in SUPPORTED:
            return key
        if key in _ALIASES:
            return _ALIASES[key]
        try:
            return dtype_name(np.dtype(key))
        except TypeError:
            raise ValueError(f"unsupported dtype {spec!r}; expected one of {SUPPORTED}") from None
    if spec is str:
        return "string"
    if spec is int:
        return "int64"
    if spec is float:
        return "float64"
    mod = type(spec).__module__
    if mod.startswith("torch"):
        name = str(spec).replace("torch.", "")
        if name in _NUMPY_TO_NAME:
            return _NUMPY_TO_NAME[name]
        raise ValueError(f"unsupported torch dtype {spec!r}")
    try:
        np_dtype = np.dtype(spec)
    except TypeError:
        raise ValueError(f"unsupported dtype {spec!r}; expected one of {SUPPORTED}") from None
    if np_dtype.kind in ("U", "S", "O", "T"):
        return "string"
    if np_dtype.name in _NUMPY_TO_NAME:
        return _NUMPY_TO_NAME[np_dtype.name]
    raise ValueError(f"unsupported dtype {np_dtype!r}; expected one of {SUPPORTED}")


def normalize_schema(schema: SchemaSpec) -> list[tuple[str, str]]:
    """Return [(name, dtype), ...] with validated, canonical dtype names."""
    items = list(schema.items()) if isinstance(schema, Mapping) else list(schema)
    if not items:
        raise ValueError("schema must not be empty")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in items:
        if isinstance(entry, str):
            raise ValueError("schema entries must be (name, dtype) pairs or a {name: dtype} mapping")
        name, dtype = entry
        name = str(name)
        if name in seen:
            raise ValueError(f"duplicate column name in schema: {name!r}")
        seen.add(name)
        out.append((name, dtype_name(dtype)))
    return out


def numpy_dtype(name: str) -> np.dtype:
    if name not in NUMERIC:
        raise ValueError(f"{name!r} is not a numeric dtype")
    return np.dtype(name)


def common_numeric_dtype(schema: Sequence[tuple[str, str]]) -> str:
    """Smallest supported dtype that holds every numeric column of the schema."""
    names = [dtype for _, dtype in schema if dtype in NUMERIC]
    if not names:
        raise ValueError("schema has no numeric columns")
    return dtype_name(np.result_type(*[np.dtype(n) for n in names]))


def _parse_kind(field: str) -> str:
    if field == "":
        return "empty"
    try:
        int(field)
        return "int64"
    except ValueError:
        pass
    try:
        float(field)
        return "float64"
    except ValueError:
        return "string"


def infer_schema_from_text(head: bytes, *, has_header: bool | None, delimiter: str,
                           sample_rows: int = 200) -> tuple[list[tuple[str, str]], bool]:
    """Infer (schema, has_header) from the first bytes of a CSV file.

    The header is detected when the first line has at least one field that does
    not parse as a number. Column dtypes widen across sampled rows:
    int64 -> float64 -> string. Empty fields count as float64 (NaN).
    """
    text = head.decode("utf-8", errors="replace")
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    if head.endswith(b"\n") or len(lines) > 1:
        lines = lines[:-1]  # the last piece may be a partial row
    lines = [ln for ln in lines if ln != ""]
    if not lines:
        raise ValueError("cannot infer a schema from an empty file")
    first = lines[0].split(delimiter)
    if has_header is None:
        has_header = any(_parse_kind(f) == "string" for f in first)
    names = first if has_header else [f"c{i}" for i in range(len(first))]
    rank = {"empty": 0, "int64": 1, "float64": 2, "string": 3}
    kinds = ["empty"] * len(names)
    data_lines = lines[1:] if has_header else lines
    for ln in data_lines[:sample_rows]:
        fields = ln.split(delimiter)
        if len(fields) != len(names):
            raise ValueError(f"inconsistent column count while inferring schema: expected {len(names)}, got {len(fields)}")
        for i, f in enumerate(fields):
            k = _parse_kind(f)
            if rank[k] > rank[kinds[i]]:
                kinds[i] = k
    schema = [(n, "float64" if k == "empty" else k) for n, k in zip(names, kinds)]
    return schema, has_header
