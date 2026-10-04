"""The `.fcsv.json` acceleration sidecar written next to canonical CSV files."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ._version import __version__

FORMAT = "fastcsv-canonical"
VERSION = 7
MIN_VERSION = 6


def sidecar_path(path: str | os.PathLike) -> Path:
    return Path(str(path) + ".fcsv.json")


def load_metadata(path: str | os.PathLike) -> dict[str, Any]:
    """Load the sidecar for `path` (raises FileNotFoundError if absent)."""
    return json.loads(sidecar_path(path).read_text(encoding="utf-8"))


def read_metadata(path: str | os.PathLike) -> dict[str, Any] | None:
    p = sidecar_path(path)
    if not p.exists():
        return None
    try:
        return load_metadata(path)
    except (OSError, ValueError):
        return None


def validate_metadata(path: str | os.PathLike, meta: dict[str, Any]) -> None:
    """Cheap stale-sidecar protection; never scans the CSV contents."""
    if meta.get("format") != FORMAT:
        raise ValueError("unrecognized FastCSV sidecar format")
    version = int(meta.get("version", 0))
    if version < MIN_VERSION or version > VERSION:
        raise ValueError(f"unsupported FastCSV sidecar version: {version}")
    st = Path(path).stat()
    if meta.get("bytes") is not None and int(meta["bytes"]) != st.st_size:
        raise ValueError("FastCSV sidecar does not match CSV size; remove/regenerate the sidecar")
    if meta.get("mtime_ns") is not None and int(meta["mtime_ns"]) != st.st_mtime_ns:
        raise ValueError("FastCSV sidecar does not match CSV modification time; remove/regenerate the sidecar")
    rows = int(meta.get("rows", 0))
    counts = [int(x) for x in meta.get("chunk_row_counts", [])]
    offsets = [int(x) for x in meta.get("chunk_offsets", [])]
    if rows < 0 or (counts and sum(counts) != rows):
        raise ValueError("FastCSV sidecar has inconsistent row counts")
    if counts and len(offsets) + 1 != len(counts):
        raise ValueError("FastCSV sidecar has inconsistent chunk boundaries")


def build_metadata(path: Path, result: dict[str, Any], schema: list[tuple[str, str]], *,
                   header: bool, delimiter: str, chunk_rows: int, target_chunk_bytes: int) -> dict[str, Any]:
    st = path.stat()
    return {
        "format": FORMAT,
        "version": VERSION,
        "library_version": __version__,
        "rows": int(result["rows"]),
        "bytes": int(result["bytes_written"]),
        "mtime_ns": int(st.st_mtime_ns),
        "header": bool(header),
        "delimiter": delimiter,
        "newline": "LF",
        "quoted_fields": False,
        "quoted_newlines": False,
        "trusted_contract": "utf8-lf-no-quotes-fixed-schema",
        "chunk_rows_limit": int(chunk_rows),
        "target_chunk_bytes": int(target_chunk_bytes),
        "chunk_offsets": [int(x) for x in result["chunk_offsets"]],
        "chunk_row_counts": [int(x) for x in result["chunk_row_counts"]],
        "chunk_byte_counts": [int(x) for x in result["chunk_byte_counts"]],
        "chunk_string_bytes": [[int(v) for v in row] for row in result["chunk_string_bytes"]],
        "columns": [{"name": n, "dtype": t} for n, t in schema],
    }


def write_metadata_atomic(meta_path: Path, tmp_meta: Path, payload: dict[str, Any]) -> None:
    with tmp_meta.open("w", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_meta, meta_path)
