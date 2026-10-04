"""Zero-copy bridges between NumPy, torch, Arrow and the C++ core."""
from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ._schema import NUMERIC, dtype_name, numpy_dtype


def is_torch_tensor(obj: Any) -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(obj, torch.Tensor)


def is_pandas_series(obj: Any) -> bool:
    pd = sys.modules.get("pandas")
    return pd is not None and isinstance(obj, pd.Series)


def is_arrow_array(obj: Any) -> bool:
    pa = sys.modules.get("pyarrow")
    return pa is not None and isinstance(obj, (pa.Array, pa.ChunkedArray))


def torch_dtype(name: str):
    import torch

    return {"int64": torch.int64, "int32": torch.int32, "float64": torch.float64, "float32": torch.float32}[name]


def tensor_as_numpy(tensor) -> np.ndarray:
    """Writable NumPy view sharing memory with a CPU tensor (no copy)."""
    if tensor.device.type != "cpu":
        raise ValueError("only CPU tensors can be viewed as NumPy arrays; pass device=... to read_csv instead")
    return tensor.detach().numpy()


def numeric_source(obj: Any, dtype: str, rows: int, what: str) -> np.ndarray:
    """1-D NumPy array of `dtype` for the writer; a view when the input already matches."""
    if dtype not in NUMERIC:
        raise ValueError(f"{what}: {dtype!r} is not numeric")
    if is_torch_tensor(obj):
        obj = obj.detach()
        if obj.device.type != "cpu":
            obj = obj.cpu()
        arr = obj.numpy()
    elif is_pandas_series(obj):
        arr = obj.to_numpy()
    elif is_arrow_array(obj):
        import pyarrow as pa

        if isinstance(obj, pa.ChunkedArray):
            obj = obj.combine_chunks()
        arr = obj.to_numpy(zero_copy_only=False)
    else:
        arr = np.asarray(obj)
    if arr.ndim != 1:
        raise ValueError(f"{what}: expected a 1-D column, got shape {arr.shape}")
    if arr.shape[0] != rows:
        raise ValueError(f"{what}: has {arr.shape[0]} rows, expected {rows}")
    target = numpy_dtype(dtype)
    if arr.dtype != target:
        arr = arr.astype(target)  # the only copy, and only when dtypes differ
    if arr.strides[0] <= 0:
        arr = np.ascontiguousarray(arr)
    return arr


def _pack_arrow(arr) -> tuple[np.ndarray, np.ndarray]:
    """(uint64 offsets, uint8 data) from an Arrow string array; data is zero-copy."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks() if arr.num_chunks != 1 else arr.chunk(0)
    if pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_decode()
    if not (pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type)):
        arr = pc.cast(arr, pa.large_string())
    if arr.null_count:
        arr = pc.fill_null(arr, "")
    if arr.offset != 0:
        arr = pa.concat_arrays([arr])  # realign a sliced array
    buffers = arr.buffers()
    offset_dtype = np.int32 if pa.types.is_string(arr.type) else np.int64
    offsets = np.frombuffer(buffers[1], dtype=offset_dtype, count=len(arr) + 1)
    if buffers[2] is None:
        data = np.zeros(0, dtype=np.uint8)
    else:
        data = np.frombuffer(buffers[2], dtype=np.uint8, count=int(offsets[-1]))
    return offsets.astype(np.uint64), data


def string_source(obj: Any, rows: int, what: str) -> Any:
    """Writer source for a string column: (offsets, data) buffers when Arrow-backed, else a sequence of str."""
    if is_pandas_series(obj):
        pd = sys.modules["pandas"]
        storage = getattr(obj.dtype, "storage", None)
        if isinstance(obj.dtype, pd.ArrowDtype) or storage == "pyarrow":
            try:
                import pyarrow as pa

                return _pack_arrow(pa.Array.from_pandas(obj))
            except Exception:  # pragma: no cover - fall back to Python strings
                pass
        obj = obj.to_numpy(dtype=object)
    if is_arrow_array(obj):
        return _pack_arrow(obj)
    if isinstance(obj, tuple) and len(obj) == 2:
        offsets = np.ascontiguousarray(np.asarray(obj[0], dtype=np.uint64))
        data = None if obj[1] is None else np.ascontiguousarray(np.asarray(obj[1], dtype=np.uint8))
        if offsets.shape != (rows + 1,):
            raise ValueError(f"{what}: packed offsets must have rows+1 entries")
        return offsets, data
    if is_torch_tensor(obj):
        raise TypeError(f"{what}: tensors cannot hold strings")
    if isinstance(obj, np.ndarray):
        if obj.ndim != 1:
            raise ValueError(f"{what}: expected a 1-D column, got shape {obj.shape}")
        if obj.dtype.kind != "O":
            obj = obj.astype(object)
    elif not isinstance(obj, (list, tuple)):
        obj = list(obj)
    if len(obj) != rows:
        raise ValueError(f"{what}: has {len(obj)} rows, expected {rows}")
    return obj


def arrow_table_from_packed(columns: dict[str, Any], schema: list[tuple[str, str]]):
    """Arrow table whose string columns wrap the packed buffers without copying."""
    import pyarrow as pa

    arrays, names = [], []
    for name, dtype in schema:
        if dtype == "skip":
            continue
        names.append(name)
        if dtype != "string":
            arrays.append(pa.array(columns[name], from_pandas=False))
            continue
        packed = columns[name]
        offsets = np.asarray(packed["offsets"], dtype=np.uint64)
        data = np.asarray(packed["data"], dtype=np.uint8)
        if len(offsets) and int(offsets[-1]) >= 2**63:
            raise OverflowError("packed string arena exceeds Arrow large_string offset range")
        arrays.append(pa.Array.from_buffers(
            pa.large_string(), len(offsets) - 1,
            [None, pa.py_buffer(offsets.view(np.int64)), pa.py_buffer(data)],
        ))
    return pa.Table.from_arrays(arrays, names=names)


def packed_to_object_array(packed: dict[str, Any]) -> np.ndarray:
    from . import _core

    offsets = np.asarray(packed["offsets"], dtype=np.uint64)
    out = np.empty(max(len(offsets) - 1, 0), dtype=object)
    _core.fill_object_array(offsets, packed["data"], out)
    return out


__all__ = [
    "arrow_table_from_packed", "dtype_name", "is_arrow_array", "is_pandas_series", "is_torch_tensor",
    "numeric_source", "packed_to_object_array", "string_source", "tensor_as_numpy", "torch_dtype",
]
