"""vvtk_fastcsv: strict CSV straight into NumPy arrays and torch tensors.

The C++ core memory-maps the file, scans delimiters with runtime-dispatched
AVX2/SSE2 code, converts numbers with fast_float/std::from_chars and writes
every value directly into the final NumPy or torch allocation. Parallel reads
use a small JSON sidecar written next to the CSV, or one parallel inspection
pass for files produced elsewhere.
"""
from ._read import ReadPlan, inspect_csv, plan_read, read_csv, read_numpy, read_torch
from ._schema import SUPPORTED as SUPPORTED_DTYPES
from ._schema import normalize_schema
from ._sidecar import load_metadata
from ._version import __version__
from ._write import write_csv

try:
    from . import _core
except ImportError:  # pragma: no cover
    _core = None


def build_info() -> dict[str, object]:
    """Report which accelerations the installed extension was built with."""
    if _core is None:
        return {"cpp_extension": False, "fast_float": False, "simd_scanner": False, "simd_runtime": "scalar",
                "direct_buffers": False, "sidecar_version": 0, "version": __version__}
    return {
        "cpp_extension": True,
        "fast_float": bool(_core.has_fast_float),
        "simd_scanner": bool(_core.has_simd_scanner),
        "simd_runtime": str(_core.simd_runtime),
        "direct_buffers": True,
        "sidecar_version": int(_core.format_version),
        "version": __version__,
    }


__all__ = [
    "ReadPlan",
    "SUPPORTED_DTYPES",
    "__version__",
    "build_info",
    "inspect_csv",
    "load_metadata",
    "normalize_schema",
    "plan_read",
    "read_csv",
    "read_numpy",
    "read_torch",
    "write_csv",
]
