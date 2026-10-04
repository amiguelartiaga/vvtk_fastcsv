"""Reading strict CSV straight into NumPy arrays, torch tensors and friends."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._buffers import (
    arrow_table_from_packed,
    is_torch_tensor,
    packed_to_object_array,
    tensor_as_numpy,
    torch_dtype,
)
from ._schema import NUMERIC, common_numeric_dtype, dtype_name, infer_schema_from_text, normalize_schema
from ._sidecar import read_metadata, validate_metadata

try:
    from . import _core
except ImportError:  # pragma: no cover - source checkout without a build
    _core = None

DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
RESPLIT_MIN_BYTES = 4 * 1024 * 1024
OUTPUTS = ("numpy", "torch", "dict", "torch_dict", "packed", "pandas", "pandas_arrow", "arrow", "polars")


def _require_core():
    if _core is None:
        raise RuntimeError("vvtk_fastcsv C++ extension is not built; run `pip install .`")
    return _core


@dataclass
class ReadPlan:
    """Everything the parser needs to allocate final buffers exactly once."""

    schema: list[tuple[str, str]]
    has_header: bool
    delimiter: str
    rows: int
    chunk_offsets: list[int] = field(default_factory=list)
    chunk_row_counts: list[int] = field(default_factory=list)
    chunk_string_bytes: list[list[int]] = field(default_factory=list)
    source: str = "inspect"  # "sidecar" or "inspect"
    input_bytes: int = 0

    @property
    def chunks(self) -> int:
        return len(self.chunk_row_counts)


def _check_delimiter(delimiter: str) -> str:
    if not isinstance(delimiter, str) or len(delimiter.encode("utf-8")) != 1:
        raise ValueError("delimiter must be a single one-byte character")
    return delimiter


def _detect_header(first_line: bytes, schema: list[tuple[str, str]], delimiter: str) -> bool:
    """With a known schema, the first line is a header if a numeric column fails to parse."""
    fields = first_line.decode("utf-8", errors="replace").rstrip("\r").split(delimiter)
    if len(fields) != len(schema):
        return True
    for value, (_, dtype) in zip(fields, schema):
        if dtype in NUMERIC and value != "":
            try:
                float(value)
            except ValueError:
                return True
    return False


def inspect_csv(path: str | os.PathLike, schema, *, has_header: bool = True, delimiter: str = ",",
                threads: int = 0, target_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
                sequential_io_hint: bool = True) -> dict[str, Any]:
    """Count rows and packed string bytes per chunk without materializing values.

    This is the pass that sidecar-free files go through before parsing. It runs
    in parallel over byte-balanced, row-aligned chunks.
    """
    core = _require_core()
    return dict(core.inspect(str(path), normalize_schema(schema), bool(has_header), _check_delimiter(delimiter),
                             int(threads), int(target_chunk_bytes), bool(sequential_io_hint)))


def plan_read(path: str | os.PathLike, schema=None, *, has_header: bool | None = None,
              delimiter: str | None = None, threads: int = 0, validate: bool = True,
              target_chunk_bytes: int = DEFAULT_CHUNK_BYTES, sequential_io_hint: bool = True) -> ReadPlan:
    """Resolve schema, header, delimiter and the chunk plan for a file.

    A valid sidecar supplies the plan directly. Otherwise the schema is inferred
    from the first rows when not given, and one parallel inspection pass
    produces the plan.
    """
    core = _require_core()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    meta = read_metadata(path)
    if meta is not None and validate:
        validate_metadata(path, meta)
    user_schema = normalize_schema(schema) if schema is not None else None

    if meta is not None:
        meta_schema = [(str(c["name"]), dtype_name(c["dtype"])) for c in meta["columns"]]
        meta_header = bool(meta.get("header", True))
        meta_delim = str(meta.get("delimiter", ","))
        resolved = user_schema or meta_schema
        if len(resolved) != len(meta_schema):
            raise ValueError(f"schema has {len(resolved)} columns but the file has {len(meta_schema)}")
        if has_header is None:
            has_header = meta_header
        if delimiter is None:
            delimiter = meta_delim
        delimiter = _check_delimiter(delimiter)
        # The sidecar plan is reusable when every requested string column was
        # also a string column when the file was written (its per-chunk byte
        # counts exist); string columns may be re-read as 'skip' or numeric.
        meta_string_ord = {i: ord for ord, i in enumerate(i for i, (_, t) in enumerate(meta_schema) if t == "string")}
        wanted = [i for i, (_, t) in enumerate(resolved) if t == "string"]
        reusable = all(i in meta_string_ord for i in wanted)
        if has_header == meta_header and delimiter == meta_delim and reusable:
            picks = [meta_string_ord[i] for i in wanted]
            plan = ReadPlan(
                schema=resolved, has_header=has_header, delimiter=delimiter, rows=int(meta.get("rows", 0)),
                chunk_offsets=[int(x) for x in meta.get("chunk_offsets", [])],
                chunk_row_counts=[int(x) for x in meta.get("chunk_row_counts", [])],
                chunk_string_bytes=[[int(row[k]) for k in picks] for row in meta.get("chunk_string_bytes", [])],
                source="sidecar", input_bytes=int(meta.get("bytes", path.stat().st_size)),
            )
            # A coarse sidecar (e.g. written with large chunks) would leave
            # workers idle; one cheap parallel inspection re-splits the file.
            workers = int(threads) or (os.cpu_count() or 1)
            scans_fields = bool(wanted)
            too_few = plan.chunks * 2 <= workers if scans_fields else plan.chunks < workers
            if too_few and plan.input_bytes >= RESPLIT_MIN_BYTES and plan.rows > 0:
                target_chunk_bytes = max(1024 * 1024, plan.input_bytes // (2 * workers))
            else:
                return plan
        user_schema = resolved

    if delimiter is None:
        delimiter = ","
    delimiter = _check_delimiter(delimiter)
    if user_schema is None or has_header is None:
        if user_schema is None:
            with path.open("rb") as f:
                head = f.read(256 * 1024)
            if not head.strip():
                raise ValueError("cannot infer a schema from an empty file; pass schema=")
            user_schema, has_header = infer_schema_from_text(head, has_header=has_header, delimiter=delimiter)
        else:
            has_header = _detect_header(core.first_line(str(path)), user_schema, delimiter)

    info = core.inspect(str(path), user_schema, bool(has_header), delimiter, int(threads),
                        int(target_chunk_bytes), bool(sequential_io_hint))
    return ReadPlan(
        schema=user_schema, has_header=bool(has_header), delimiter=delimiter, rows=int(info["rows"]),
        chunk_offsets=[int(x) for x in info["chunk_offsets"]],
        chunk_row_counts=[int(x) for x in info["chunk_row_counts"]],
        chunk_string_bytes=[[int(v) for v in row] for row in info["chunk_string_bytes"]],
        source="inspect", input_bytes=int(info["input_bytes"]),
    )


def _check_out_numpy(out: np.ndarray, rows: int, ncols: int, dtype: str) -> np.ndarray:
    if not isinstance(out, np.ndarray):
        raise TypeError("out must be a NumPy array for output='numpy'")
    if out.ndim != 2 or out.shape[1] != ncols or out.shape[0] < rows:
        raise ValueError(f"out must have shape (>= {rows}, {ncols}); got {out.shape}")
    if out.dtype != np.dtype(dtype):
        raise ValueError(f"out has dtype {out.dtype}, expected {dtype} (pass dtype= to parse directly as {out.dtype})")
    view = out[:rows]
    if not view.flags.c_contiguous or not view.flags.writeable:
        raise ValueError("out must be a writable C-contiguous (row-major) array")
    return view


def _check_out_torch(out, rows: int, ncols: int, dtype: str):
    if not is_torch_tensor(out):
        raise TypeError("out must be a torch.Tensor for output='torch'")
    if out.dim() != 2 or out.shape[1] != ncols or out.shape[0] < rows:
        raise ValueError(f"out must have shape (>= {rows}, {ncols}); got {tuple(out.shape)}")
    if out.dtype != torch_dtype(dtype):
        raise ValueError(f"out has dtype {out.dtype}, expected {torch_dtype(dtype)} (pass dtype= to override)")
    view = out[:rows]
    if not view.is_contiguous():
        raise ValueError("out must be a contiguous (row-major) tensor")
    return view


def _read_matrix(path: Path, plan: ReadPlan, output: str, dtype, out, device, pin_memory: bool,
                 threads: int, sequential_io_hint: bool, empty_float_is_nan: bool):
    core = _require_core()
    active = [(n, t) for n, t in plan.schema if t != "skip"]
    if any(t == "string" for _, t in active):
        names = [n for n, t in active if t == "string"]
        raise ValueError(
            f"output={output!r} needs numeric columns, but {names} are strings; pass a schema marking them "
            "'skip' (or pass dtype= to parse them numerically), or use output='dict'/'pandas'"
        )
    mdtype = dtype_name(dtype) if dtype is not None else common_numeric_dtype(plan.schema)
    if mdtype not in NUMERIC:
        raise ValueError(f"matrix dtype must be numeric, got {mdtype!r}")
    rows, ncols = plan.rows, len(active)
    override = [(n, "skip" if t == "skip" else mdtype) for n, t in plan.schema]
    has_skip = ncols != len(plan.schema)

    finish = None
    if output == "numpy":
        view = _check_out_numpy(out, rows, ncols, mdtype) if out is not None else np.empty((rows, ncols), dtype=mdtype)
        np_view = view
        result = view
    else:
        import torch

        tdtype = torch_dtype(mdtype)
        if device is None:
            dev = out.device if out is not None else torch.device("cpu")
        else:
            dev = torch.device(device)
        if out is not None:
            out = _check_out_torch(out, rows, ncols, mdtype)
        if dev.type == "cpu":
            pin = bool(pin_memory) and torch.cuda.is_available()
            tensor = out if out is not None else torch.empty((rows, ncols), dtype=tdtype, pin_memory=pin)
            np_view = tensor_as_numpy(tensor)
            result = tensor
        else:
            # Parse into (pinned) host memory, then one asynchronous device copy.
            pin = dev.type == "cuda" and torch.cuda.is_available()
            staging = torch.empty((rows, ncols), dtype=tdtype, pin_memory=pin)
            np_view = tensor_as_numpy(staging)

            def finish():
                if out is not None:
                    out.copy_(staging, non_blocking=True)
                    return out
                return staging.to(dev, non_blocking=True)

            result = None

    common = dict(has_header=plan.has_header, delimiter=plan.delimiter, expected_rows=rows,
                  chunk_offsets=plan.chunk_offsets, chunk_row_counts=plan.chunk_row_counts,
                  threads=int(threads), sequential_io_hint=bool(sequential_io_hint),
                  empty_float_is_nan=bool(empty_float_is_nan))
    if not has_skip and np_view.flags.c_contiguous:
        stats = core.read_matrix_into(str(path), np_view, names=[n for n, _ in active], **common)
    else:
        targets: list[Any] = []
        j = 0
        for _, t in plan.schema:
            if t == "skip":
                targets.append(None)
            else:
                targets.append(np_view[:, j])
                j += 1
        stats = core.read_into(str(path), override, targets, chunk_string_bytes=[], **common)
    if finish is not None:
        result = finish()
    return result, dict(stats)


def _read_columns(path: Path, plan: ReadPlan, output: str, strings: str, threads: int,
                  sequential_io_hint: bool, empty_float_is_nan: bool):
    core = _require_core()
    rows = plan.rows
    torch_out = output == "torch_dict"
    if torch_out:
        import torch
    targets: list[Any] = []
    columns: dict[str, Any] = {}
    string_ord = 0
    for name, t in plan.schema:
        if t == "skip":
            targets.append(None)
        elif t == "string":
            total = sum(int(chunk[string_ord]) for chunk in plan.chunk_string_bytes)
            string_ord += 1
            offsets = np.empty(rows + 1, dtype=np.uint64)
            data = np.empty(total, dtype=np.uint8)
            targets.append((offsets, data))
            columns[name] = {"offsets": offsets, "data": data}
        elif torch_out:
            tensor = torch.empty(rows, dtype=torch_dtype(t))
            targets.append(tensor_as_numpy(tensor))
            columns[name] = tensor
        else:
            arr = np.empty(rows, dtype=t)
            targets.append(arr)
            columns[name] = arr

    stats = dict(core.read_into(
        str(path), plan.schema, targets, plan.has_header, plan.delimiter, rows,
        plan.chunk_offsets, plan.chunk_row_counts, plan.chunk_string_bytes,
        int(threads), bool(sequential_io_hint), bool(empty_float_is_nan),
    ))

    if output == "packed":
        return columns, stats
    if output in ("arrow", "polars", "pandas_arrow"):
        table = arrow_table_from_packed(columns, plan.schema)
        if output == "arrow":
            return table, stats
        if output == "polars":
            import polars as pl

            return pl.from_arrow(table), stats
        import pandas as pd

        return table.to_pandas(types_mapper=pd.ArrowDtype), stats

    for name, t in plan.schema:
        if t == "string" and strings != "packed":
            decoded = packed_to_object_array(columns[name])
            columns[name] = decoded.tolist() if torch_out else decoded
    if output in ("dict", "torch_dict"):
        return columns, stats
    if output == "pandas":
        import pandas as pd

        return pd.DataFrame(columns, copy=False), stats
    raise ValueError(f"output must be one of {OUTPUTS}, got {output!r}")


def read_csv(path: str | os.PathLike, schema=None, *, output: str = "numpy", dtype=None, out=None,
             device=None, pin_memory: bool = False, has_header: bool | None = None,
             delimiter: str | None = None, threads: int = 0, strings: str = "python",
             empty_float_is_nan: bool = True, validate_metadata: bool = True,
             sequential_io_hint: bool = True, return_stats: bool = False):
    """Read a strict CSV through the C++ core, parsing straight into the final memory.

    Parameters
    ----------
    path:
        CSV file. A ``<path>.fcsv.json`` sidecar written by :func:`write_csv`
        is used automatically; otherwise one inspection pass plans the read.
    schema:
        ``{name: dtype}`` or ``[(name, dtype), ...]`` with dtypes ``int64``,
        ``int32``, ``float64``, ``float32``, ``string`` or ``skip`` (NumPy and
        torch dtypes are accepted too). Inferred from the first rows when omitted
        and no sidecar exists.
    output:
        ``"numpy"`` (default) or ``"torch"`` return one row-major 2-D matrix of
        all non-skipped columns, parsed directly into that allocation.
        ``"dict"`` / ``"torch_dict"`` return per-column 1-D arrays / tensors
        (string columns as object arrays / lists). ``"pandas"``,
        ``"pandas_arrow"``, ``"arrow"``, ``"polars"`` build those containers
        with the string columns wrapped without per-row Python objects where
        the container allows it. ``"packed"`` exposes the raw parser storage.
    dtype:
        Matrix element dtype for ``numpy``/``torch`` outputs. Every column is
        parsed directly as this dtype; defaults to the common numeric type.
    out:
        Preallocated 2-D array/tensor of shape ``(>= rows, ncols)`` to parse
        into (``numpy``/``torch`` outputs). Returns ``out[:rows]``.
    device, pin_memory:
        For ``output="torch"``: target device (``"cuda"``, ...). Non-CPU
        targets are parsed into pinned host memory and copied once,
        asynchronously. ``pin_memory`` pins a CPU result for later transfers.
    threads:
        Worker threads; ``0`` means hardware concurrency.
    empty_float_is_nan:
        Empty float cells (as pandas writes NaN) parse as NaN instead of failing.
    return_stats:
        Also return a dict with rows, chunks, workers and the plan source.
    """
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}, got {output!r}")
    if strings not in ("python", "packed"):
        raise ValueError("strings must be 'python' or 'packed'")
    path = Path(path)
    plan = plan_read(path, schema, has_header=has_header, delimiter=delimiter, threads=threads,
                     validate=validate_metadata, sequential_io_hint=sequential_io_hint)
    if output in ("numpy", "torch"):
        result, stats = _read_matrix(path, plan, output, dtype, out, device, pin_memory, threads,
                                     sequential_io_hint, empty_float_is_nan)
    else:
        if dtype is not None or out is not None or device is not None:
            raise ValueError("dtype=, out= and device= apply to output='numpy' or output='torch'")
        result, stats = _read_columns(path, plan, output, strings, threads, sequential_io_hint, empty_float_is_nan)
    stats.update(plan=plan.source, schema=list(plan.schema), has_header=plan.has_header)
    return (result, stats) if return_stats else result


def read_numpy(path: str | os.PathLike, dtype=None, **kwargs) -> np.ndarray:
    """Read a numeric CSV into one row-major NumPy matrix. See :func:`read_csv`."""
    return read_csv(path, output="numpy", dtype=dtype, **kwargs)


def read_torch(path: str | os.PathLike, dtype=None, device=None, pin_memory: bool = False, **kwargs):
    """Read a numeric CSV into one row-major torch tensor. See :func:`read_csv`."""
    return read_csv(path, output="torch", dtype=dtype, device=device, pin_memory=pin_memory, **kwargs)
