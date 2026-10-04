"""Writing canonical CSV (plus sidecar) from NumPy, torch, pandas, Arrow and polars data."""
from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ._buffers import is_arrow_array, is_pandas_series, is_torch_tensor, numeric_source, string_source
from ._schema import dtype_name, normalize_schema
from ._sidecar import build_metadata, sidecar_path, write_metadata_atomic

try:
    from . import _core
except ImportError:  # pragma: no cover
    _core = None

DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024


def _is_pandas_frame(obj: Any) -> bool:
    pd = sys.modules.get("pandas")
    return pd is not None and isinstance(obj, pd.DataFrame)


def _is_arrow_table(obj: Any) -> bool:
    pa = sys.modules.get("pyarrow")
    return pa is not None and isinstance(obj, pa.Table)


def _is_polars_frame(obj: Any) -> bool:
    pl = sys.modules.get("polars")
    return pl is not None and isinstance(obj, pl.DataFrame)


def _matrix_columns(arr, names) -> tuple[list[str], list[Any], int]:
    if arr.ndim == 1:
        arr = arr.reshape(arr.shape[0], 1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got {arr.ndim}-D")
    rows, ncols = int(arr.shape[0]), int(arr.shape[1])
    if names is None:
        names = [f"c{i}" for i in range(ncols)]
    names = [str(n) for n in names]
    if len(names) != ncols:
        raise ValueError(f"columns has {len(names)} names but the matrix has {ncols} columns")
    # Column views share the matrix memory: the writer reads them with a stride.
    return names, [arr[:, j] for j in range(ncols)], rows


def _columns_from(data: Any, columns) -> tuple[list[str], list[Any], int]:
    if is_torch_tensor(data):
        data = data.detach()
        if data.device.type != "cpu":
            data = data.cpu()
        return _matrix_columns(data.numpy(), columns)
    if isinstance(data, np.ndarray):
        if data.dtype.kind == "O" and data.ndim == 1:
            return ([str(columns[0]) if columns else "c0"], [data], len(data))
        return _matrix_columns(data, columns)
    if _is_pandas_frame(data):
        names = [str(c) for c in data.columns]
        if len(set(names)) != len(names):
            raise ValueError("DataFrame has duplicate column names")
        return names, [data[c] for c in data.columns], len(data)
    if _is_polars_frame(data):
        data = data.to_arrow()
    if _is_arrow_table(data):
        names = list(data.column_names)
        return names, [data.column(i) for i in range(data.num_columns)], data.num_rows
    if isinstance(data, Mapping):
        names = [str(k) for k in data]
        values = list(data.values())
        if not values:
            raise ValueError("data must have at least one column")
        first = values[0]
        rows = int(first.shape[0]) if hasattr(first, "shape") else len(first)
        return names, values, rows
    raise TypeError(
        "data must be a 2-D NumPy array or torch tensor, a {name: column} mapping, "
        "a pandas/polars DataFrame or a pyarrow Table"
    )


def _infer_dtype(obj: Any) -> str:
    if is_torch_tensor(obj):
        return dtype_name(obj.dtype)
    if is_pandas_series(obj):
        pd = sys.modules["pandas"]
        if pd.api.types.is_integer_dtype(obj.dtype):
            return "int32" if getattr(obj.dtype, "itemsize", 8) <= 4 else "int64"
        if pd.api.types.is_float_dtype(obj.dtype):
            return "float32" if getattr(obj.dtype, "itemsize", 8) <= 4 else "float64"
        if pd.api.types.is_bool_dtype(obj.dtype):
            return "int32"
        return "string"
    if is_arrow_array(obj):
        import pyarrow as pa

        t = obj.type
        if pa.types.is_integer(t):
            return "int64" if t.bit_width > 32 else "int32"
        if pa.types.is_floating(t):
            return "float64" if t.bit_width > 32 else "float32"
        return "string"
    if isinstance(obj, tuple) and len(obj) == 2:
        return "string"  # packed (offsets, data)
    arr = obj if isinstance(obj, np.ndarray) else np.asarray(obj)
    return dtype_name(arr.dtype)


def _resolve_schema(names: list[str], values: list[Any], schema) -> list[tuple[str, str]]:
    if schema is None:
        return [(n, _infer_dtype(v)) for n, v in zip(names, values)]
    given = dict(normalize_schema(schema))
    unknown = [n for n in given if n not in names]
    if unknown:
        raise ValueError(f"schema names {unknown} are not columns of the data {names}")
    missing = [n for n in names if n not in given]
    if missing:
        raise ValueError(f"schema is missing dtypes for columns {missing}")
    out = [(n, given[n]) for n in names]
    if any(t == "skip" for _, t in out):
        raise ValueError("'skip' columns cannot be written; drop them from the data instead")
    return out


def write_csv(data: Any, path: str | os.PathLike, *, schema=None, columns=None, header: bool = True,
              delimiter: str = ",", metadata: bool = True, chunk_rows: int = 0,
              target_chunk_bytes: int = DEFAULT_CHUNK_BYTES, threads: int = 0, block_rows: int = 0,
              float_precision: int | None = None) -> Path:
    """Write canonical CSV (LF, no quoting) plus an acceleration sidecar.

    Parameters
    ----------
    data:
        A 2-D NumPy array or torch tensor (row-major memory is written directly,
        no transposition), a ``{name: column}`` mapping of arrays/tensors/lists,
        a pandas or polars DataFrame, or a pyarrow Table.
    path:
        Destination. The file and its ``<path>.fcsv.json`` sidecar are published
        atomically; a stale sidecar is never left behind.
    schema:
        Optional ``{name: dtype}`` overriding inferred column dtypes.
    columns:
        Column names for matrix input (default ``c0, c1, ...``).
    threads:
        Formatting workers; ``0`` means hardware concurrency. Output order is
        always preserved.
    chunk_rows, target_chunk_bytes:
        Sidecar chunking. Byte-targeted chunks (default 4 MiB) balance parallel
        reads well; ``chunk_rows`` closes a chunk exactly every N rows instead.
    float_precision:
        Significant digits for floats; ``None`` writes the shortest text that
        round-trips exactly.
    """
    if _core is None:
        raise RuntimeError("vvtk_fastcsv C++ extension is not built; run `pip install .`")
    path = Path(path)
    if len(delimiter.encode("utf-8")) != 1:
        raise ValueError("delimiter must be a single one-byte character")
    names, values, rows = _columns_from(data, columns)
    resolved = _resolve_schema(names, values, schema)
    for name in names:
        if delimiter in name or "\n" in name or "\r" in name:
            raise ValueError(f"column name {name!r} contains the delimiter or a newline")

    sources: list[Any] = []
    for (name, dtype), value in zip(resolved, values):
        what = f"column {name!r}"
        sources.append(string_source(value, rows, what) if dtype == "string" else numeric_source(value, dtype, rows, what))

    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    tmp_csv = path.with_name(f".{path.name}.fastcsv-tmp-{token}")
    meta_path = sidecar_path(path)
    tmp_meta = meta_path.with_name(f".{meta_path.name}.fastcsv-tmp-{token}")
    try:
        result = _core.write(
            str(tmp_csv), resolved, sources, int(rows), delimiter, bool(header), int(chunk_rows),
            int(target_chunk_bytes), int(threads), int(block_rows), -1 if float_precision is None else int(float_precision),
        )
        # Never leave old acceleration metadata pointing at a replacement file.
        try:
            meta_path.unlink()
        except FileNotFoundError:
            pass
        os.replace(tmp_csv, path)
        if metadata:
            payload = build_metadata(path, result, resolved, header=header, delimiter=delimiter,
                                     chunk_rows=chunk_rows, target_chunk_bytes=target_chunk_bytes)
            write_metadata_atomic(meta_path, tmp_meta, payload)
        return path
    finally:
        for leftover in (tmp_csv, tmp_meta):
            try:
                leftover.unlink()
            except FileNotFoundError:
                pass
