#!/usr/bin/env python3
"""Benchmark vvtk_fastcsv against the usual NumPy / pandas / PyArrow / polars readers and writers.

Examples
--------
    python benchmarks/benchmark.py --scale 0.1 --repeats 2          # quick smoke run
    python benchmarks/benchmark.py --mode read --threads 8           # read-only, 8 workers
    python benchmarks/benchmark.py --case tall_f32 --no-sidecar      # foreign-file path only
"""
from __future__ import annotations

import argparse
import gc
import io
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import vvtk_fastcsv as fc

CASES = {
    "tall_f64": ("Tall float64 matrix", 2_000_000, 4, "f64"),
    "tall_f32": ("Tall float32 matrix", 2_000_000, 4, "f32"),
    "wide_f64": ("Wide float64 matrix", 100_000, 64, "f64"),
    "int64": ("int64 matrix", 2_000_000, 6, "i64"),
    "mixed": ("int/float/string columns", 1_000_000, 6, "mixed"),
    "strings": ("string heavy", 500_000, 6, "strings"),
}


def make_case(rows: int, cols: int, kind: str):
    rng = np.random.default_rng(12345)
    if kind == "f64":
        return rng.random((rows, cols)), None
    if kind == "f32":
        return rng.random((rows, cols), dtype=np.float32), None
    if kind == "i64":
        return rng.integers(-10**9, 10**9, size=(rows, cols), dtype=np.int64), None
    words = np.array([f"token{i:05d}" for i in range(10_000)], dtype=object)
    data = {}
    for c in range(cols):
        name = f"c{c}"
        if kind == "mixed" and c % 3 == 0:
            data[name] = rng.integers(0, 1_000_000, size=rows, dtype=np.int64)
        elif kind == "mixed" and c % 3 == 1:
            data[name] = rng.random(rows)
        else:
            data[name] = words[(np.arange(rows) * (c + 7)) % len(words)]
    return None, data


def timed(fn, repeats: int):
    samples = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def table(headers, rows):
    text = [[str(x) for x in headers]] + [[str(x) for x in r] for r in rows]
    widths = [max(len(row[i]) for row in text) for i in range(len(headers))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = lambda row: "|" + "|".join(f" {row[i]:<{widths[i]}} " for i in range(len(widths))) + "|"  # noqa: E731
    return "\n".join([sep, fmt(text[0]), sep] + [fmt(r) for r in text[1:]] + [sep])


def optional(name):
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=[*CASES, "all"], default="all")
    ap.add_argument("--mode", choices=["read", "write", "both"], default="both")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply row counts by this factor")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="vvtk_fastcsv workers; 0 = hardware concurrency")
    ap.add_argument("--chunk-mib", type=int, default=32)
    ap.add_argument("--no-sidecar", action="store_true", help="delete the sidecar so reads go through inspection")
    ap.add_argument("--skip", default="", help="comma-separated competitor names to skip (pandas,pyarrow,polars,numpy,torch)")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))

    pd = None if "pandas" in skip else optional("pandas")
    pa_csv = None if "pyarrow" in skip else optional("pyarrow.csv")
    pl = None if "polars" in skip else optional("polars")
    torch = None if "torch" in skip else optional("torch")

    print(f"vvtk_fastcsv build: {fc.build_info()}")
    print(f"threads: {args.threads} (0=auto, {os.cpu_count()} CPUs); repeats: {args.repeats}; "
          f"sidecar: {'no' if args.no_sidecar else 'yes'}")
    selected = CASES.items() if args.case == "all" else [(args.case, CASES[args.case])]
    read_rows, write_rows = [], []

    with tempfile.TemporaryDirectory(prefix="vvtk-fastcsv-bench-") as tmp:
        root = Path(tmp)
        for key, (label, base_rows, cols, kind) in selected:
            rows = max(1, int(base_rows * args.scale))
            matrix, columns = make_case(rows, cols, kind)
            path = root / f"{key}.csv"
            data = matrix if matrix is not None else columns
            chunk = args.chunk_mib * 1024 * 1024
            fc.write_csv(data, path, target_chunk_bytes=chunk, threads=args.threads)
            mib = path.stat().st_size / 2**20
            shape = f"{rows:,}x{cols}"
            if args.no_sidecar:
                Path(str(path) + ".fcsv.json").unlink()

            if args.mode in ("write", "both"):
                results = [label, shape, f"{mib:.0f}"]
                t = timed(lambda: fc.write_csv(data, root / "w_fc.csv", target_chunk_bytes=chunk, threads=args.threads),
                          args.repeats)
                results += [f"{t:.3f}", f"{mib / t:.0f}"]
                if matrix is not None:
                    t = timed(lambda: np.savetxt(root / "w_np.csv", matrix, delimiter=",", fmt="%.17g" if matrix.dtype.kind == "f" else "%d"),
                              max(1, args.repeats // 3))
                    results += [f"{t:.3f}"]
                else:
                    results += ["-"]
                if pd is not None:
                    df = pd.DataFrame(matrix if matrix is not None else columns)
                    t = timed(lambda: df.to_csv(root / "w_pd.csv", index=False, lineterminator="\n"), args.repeats)
                    results += [f"{t:.3f}"]
                else:
                    results += ["-"]
                if pl is not None:
                    frame = pl.DataFrame(matrix if matrix is not None else columns)
                    t = timed(lambda: frame.write_csv(root / "w_pl.csv"), args.repeats)
                    results += [f"{t:.3f}"]
                else:
                    results += ["-"]
                write_rows.append(results)

            if args.mode in ("read", "both"):
                results = [label, shape, f"{mib:.0f}"]
                if matrix is not None:
                    t = timed(lambda: fc.read_csv(path, output="numpy", threads=args.threads), args.repeats)
                    results += [f"{t:.3f}", f"{mib / t:.0f}"]
                    if torch is not None:
                        t = timed(lambda: fc.read_csv(path, output="torch", threads=args.threads), args.repeats)
                        results += [f"{t:.3f}"]
                    else:
                        results += ["-"]
                    t = timed(lambda: np.loadtxt(path, delimiter=",", skiprows=1, dtype=matrix.dtype),
                              max(1, args.repeats // 3))
                    results += [f"{t:.3f}"]
                else:
                    t = timed(lambda: fc.read_csv(path, output="packed", threads=args.threads), args.repeats)
                    results += [f"{t:.3f}", f"{mib / t:.0f}"]
                    t = timed(lambda: fc.read_csv(path, output="dict", threads=args.threads), args.repeats)
                    results += [f"{t:.3f}", "-"]
                if pd is not None:
                    t = timed(lambda: pd.read_csv(path), args.repeats)
                    results += [f"{t:.3f}"]
                    if matrix is None:
                        t = timed(lambda: fc.read_csv(path, output="pandas", threads=args.threads), args.repeats)
                        results += [f"{t:.3f}"]
                    else:
                        results += ["-"]
                else:
                    results += ["-", "-"]
                if pa_csv is not None:
                    t = timed(lambda: pa_csv.read_csv(path), args.repeats)
                    results += [f"{t:.3f}"]
                else:
                    results += ["-"]
                if pl is not None:
                    t = timed(lambda: pl.read_csv(path), args.repeats)
                    results += [f"{t:.3f}"]
                else:
                    results += ["-"]
                read_rows.append(results)
            del data, matrix, columns
            gc.collect()

    if write_rows:
        print("\nWRITE (seconds, median; vvtk_fastcsv includes the sidecar)")
        print(table(["Case", "Shape", "MiB", "fastcsv s", "fastcsv MiB/s", "np.savetxt s", "pandas s", "polars s"],
                    write_rows))
    if read_rows:
        print("\nREAD (seconds, median)")
        print(table(["Case", "Shape", "MiB", "fastcsv numpy|packed s", "MiB/s", "fastcsv torch|dict s",
                     "np.loadtxt s", "pandas s", "fastcsv->pandas s", "pyarrow s", "polars s"], read_rows))
        print("\nMatrix cases: 'fastcsv numpy' parses directly into the final 2-D array; 'fastcsv torch' into a tensor.")
        print("Column cases: 'packed' is the raw parser output; 'dict' materializes Python str objects.")


if __name__ == "__main__":
    main()
