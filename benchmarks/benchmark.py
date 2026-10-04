#!/usr/bin/env python3
"""Benchmark vvtk_fastcsv against common CSV readers and writers.

Every reader in the matrix tables is asked for the same thing: a 2-D NumPy
array (so pandas/polars/pyarrow/duckdb timings include their `.to_numpy()`
step). String tables compare the containers each library produces natively.

Examples
--------
    python benchmarks/benchmark.py --quick                  # ~1 minute smoke run
    python benchmarks/benchmark.py --out benchmarks/RESULTS.md
    python benchmarks/benchmark.py --section read --cases f64_tall,i64_tall
    python benchmarks/benchmark.py --section threads --threads 1,2,4,8
"""
from __future__ import annotations

import argparse
import gc
import importlib
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import vvtk_fastcsv as fc

# key: (label, rows, cols, kind)
CASES = {
    "f64_narrow": ("float64, 2 columns", 2_000_000, 2, "f64"),
    "f64_tall": ("float64, 8 columns", 1_000_000, 8, "f64"),
    "f64_wide": ("float64, 128 columns", 50_000, 128, "f64"),
    "f32_tall": ("float32, 8 columns", 1_000_000, 8, "f32"),
    "i64_tall": ("int64, 8 columns", 1_000_000, 8, "i64"),
    "i32_tall": ("int32, 8 columns", 1_000_000, 8, "i32"),
    "mixed_numeric": ("int64 / float32 / float64, 9 columns", 1_000_000, 9, "mixed_numeric"),
    "mixed": ("int64 / float64 / string, 6 columns", 1_000_000, 6, "mixed"),
    "strings_short": ("short strings (10 B), 6 columns", 500_000, 6, "strings_short"),
    "strings_long": ("long strings (45 B), 4 columns", 250_000, 4, "strings_long"),
}
SIZE_ROWS = [100_000, 1_000_000, 5_000_000]       # float64 x 8: ~15, ~150, ~750 MiB
THREADS = [1, 2, 4, 8, 16]
MATRIX_KINDS = {"f64", "f32", "i64", "i32", "mixed_numeric"}


# --------------------------------------------------------------------------- data
def make_case(rows: int, cols: int, kind: str):
    """Returns (matrix, columns): one of them is None."""
    rng = np.random.default_rng(12345)
    if kind == "f64":
        return rng.random((rows, cols)), None
    if kind == "f32":
        return rng.random((rows, cols), dtype=np.float32), None
    if kind == "i64":
        return rng.integers(-10**12, 10**12, size=(rows, cols), dtype=np.int64), None
    if kind == "i32":
        return rng.integers(-10**9, 10**9, size=(rows, cols), dtype=np.int32), None
    words = np.array([f"token{i:05d}" for i in range(10_000)], dtype=object)
    data = {}
    for c in range(cols):
        name = f"c{c}"
        if kind == "mixed_numeric":
            if c % 3 == 0:
                data[name] = rng.integers(-10**9, 10**9, size=rows, dtype=np.int64)
            elif c % 3 == 1:
                data[name] = rng.random(rows, dtype=np.float32)
            else:
                data[name] = rng.random(rows)
        elif kind == "mixed":
            if c % 3 == 0:
                data[name] = rng.integers(0, 1_000_000, size=rows, dtype=np.int64)
            elif c % 3 == 1:
                data[name] = rng.random(rows)
            else:
                data[name] = words[(np.arange(rows) * (c + 7)) % len(words)]
        elif kind == "strings_short":
            data[name] = words[(np.arange(rows) * (c + 7)) % len(words)]
        elif kind == "strings_long":
            base = words[(np.arange(rows) * (c + 11)) % len(words)]
            data[name] = np.array([f"{x}_abcdefghijklmnopqrstuvwxyz_{i % 1000:03d}" for i, x in enumerate(base)],
                                  dtype=object)
        else:
            raise ValueError(kind)
    return None, data


# --------------------------------------------------------------------------- helpers
def optional(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def timed(fn, repeats: int, budget_s: float = 20.0) -> float:
    """Median wall time; stops early once the time budget is spent."""
    samples = []
    spent = 0.0
    for _ in range(max(1, repeats)):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        samples.append(dt)
        spent += dt
        if spent > budget_s:
            break
    return statistics.median(samples)


def fmt(t: float | None) -> str:
    return "-" if t is None else f"{t:.3f}"


def speed(mib: float, t: float | None) -> str:
    return "-" if t is None else f"{mib / t:,.0f}"


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def table_to_numpy_arrow(table):
    return np.column_stack([col.to_numpy() for col in table.columns])


class Competitors:
    def __init__(self, skip: set[str]):
        self.pd = None if "pandas" in skip else optional("pandas")
        self.pacsv = None if "pyarrow" in skip else optional("pyarrow.csv")
        self.pl = None if "polars" in skip else optional("polars")
        self.duckdb = None if "duckdb" in skip else optional("duckdb")
        self.torch = None if "torch" in skip else optional("torch")

    def versions(self) -> str:
        parts = [f"numpy {np.__version__}", f"vvtk_fastcsv {fc.__version__}"]
        for name, mod in (("pandas", self.pd), ("pyarrow", optional("pyarrow") if self.pacsv else None),
                          ("polars", self.pl), ("duckdb", self.duckdb), ("torch", self.torch)):
            if mod is not None:
                parts.append(f"{name} {mod.__version__}")
        return ", ".join(parts)

    # ---- readers that return a 2-D numpy array
    def matrix_readers(self, path: Path, nosidecar: Path, dtype, threads: int):
        readers = {
            "fastcsv numpy": lambda: fc.read_csv(path, threads=threads),
            "fastcsv numpy (no sidecar)": lambda: fc.read_csv(nosidecar, threads=threads),
        }
        if self.torch is not None:
            readers["fastcsv torch"] = lambda: fc.read_csv(path, output="torch", threads=threads)
        readers["np.loadtxt"] = lambda: np.loadtxt(path, delimiter=",", skiprows=1, dtype=dtype)
        if self.pd is not None:
            pd = self.pd
            readers["pandas (C) .to_numpy()"] = lambda: pd.read_csv(path).to_numpy()
            if self.pacsv is not None:
                readers["pandas (pyarrow) .to_numpy()"] = lambda: pd.read_csv(path, engine="pyarrow").to_numpy()
        if self.pacsv is not None:
            pacsv = self.pacsv
            readers["pyarrow.csv -> numpy"] = lambda: table_to_numpy_arrow(pacsv.read_csv(path))
        if self.pl is not None:
            pl = self.pl
            readers["polars .to_numpy()"] = lambda: pl.read_csv(path).to_numpy()
        if self.duckdb is not None:
            duckdb = self.duckdb
            readers["duckdb .fetchnumpy()"] = lambda: np.column_stack(
                list(duckdb.sql(f"SELECT * FROM read_csv('{path}')").fetchnumpy().values()))
        return readers

    # ---- readers returning each library's native container (string cases)
    def column_readers(self, path: Path, threads: int):
        readers = {
            "fastcsv packed": lambda: fc.read_csv(path, output="packed", threads=threads),
            "fastcsv dict (Python str)": lambda: fc.read_csv(path, output="dict", threads=threads),
        }
        if self.pacsv is not None:
            readers["fastcsv arrow"] = lambda: fc.read_csv(path, output="arrow", threads=threads)
        if self.pd is not None:
            pd = self.pd
            readers["fastcsv pandas"] = lambda: fc.read_csv(path, output="pandas", threads=threads)
            readers["pandas (C)"] = lambda: pd.read_csv(path)
            if self.pacsv is not None:
                readers["pandas (pyarrow)"] = lambda: pd.read_csv(path, engine="pyarrow")
        if self.pacsv is not None:
            pacsv = self.pacsv
            readers["pyarrow.csv"] = lambda: pacsv.read_csv(path)
        if self.pl is not None:
            pl = self.pl
            readers["polars"] = lambda: pl.read_csv(path)
        if self.duckdb is not None:
            duckdb = self.duckdb
            readers["duckdb .arrow()"] = lambda: duckdb.sql(f"SELECT * FROM read_csv('{path}')").arrow()
        return readers

    def writers(self, data, matrix, columns, root: Path, threads: int):
        writers = {"fastcsv": lambda: fc.write_csv(data, root / "w_fc.csv", threads=threads)}
        if matrix is not None:
            writers["np.savetxt"] = lambda: np.savetxt(root / "w_np.csv", matrix, delimiter=",",
                                                       fmt="%.17g" if matrix.dtype.kind == "f" else "%d")
        if self.pd is not None:
            df = self.pd.DataFrame(matrix if matrix is not None else columns)
            writers["pandas .to_csv()"] = lambda: df.to_csv(root / "w_pd.csv", index=False, lineterminator="\n")
        if self.pacsv is not None:
            pa = importlib.import_module("pyarrow")
            table = pa.table({f"c{i}": matrix[:, i] for i in range(matrix.shape[1])} if matrix is not None else columns)
            writers["pyarrow.csv.write_csv"] = lambda: self.pacsv.write_csv(table, root / "w_pa.csv")
        if self.pl is not None:
            frame = self.pl.DataFrame(matrix if matrix is not None else columns)
            writers["polars .write_csv()"] = lambda: frame.write_csv(root / "w_pl.csv")
        return writers


# --------------------------------------------------------------------------- sections
def write_case_files(root: Path, key: str, rows: int, cols: int, kind: str, threads: int):
    matrix, columns = make_case(rows, cols, kind)
    data = matrix if matrix is not None else columns
    path = root / f"{key}.csv"
    fc.write_csv(data, path, threads=threads)
    nosidecar = root / f"{key}_nosidecar.csv"
    nosidecar.write_bytes(path.read_bytes())
    return matrix, columns, data, path, nosidecar


def section_read(comp: Competitors, root: Path, keys, scale: float, repeats: int, threads: int):
    matrix_rows, string_rows = [], []
    for key in keys:
        label, base_rows, cols, kind = CASES[key]
        rows = max(1, int(base_rows * scale))
        matrix, columns, data, path, nosidecar = write_case_files(root, key, rows, cols, kind, threads)
        mib = path.stat().st_size / 2**20
        if kind in MATRIX_KINDS:
            dtype = matrix.dtype if matrix is not None else np.float64
            times = {name: timed(fn, repeats) for name, fn in comp.matrix_readers(path, nosidecar, dtype, threads).items()}
            matrix_rows.append((label, f"{rows:,} x {cols}", mib, times))
        else:
            times = {name: timed(fn, repeats) for name, fn in comp.column_readers(path, threads).items()}
            string_rows.append((label, f"{rows:,} x {cols}", mib, times))
        del matrix, columns, data
        gc.collect()
    out = []
    if matrix_rows:
        names = list(matrix_rows[0][3])
        out.append("### Read: numeric matrices to a NumPy array\n")
        out.append("Seconds, median. Competitor times include their conversion to a 2-D NumPy array.\n")
        out.append(md_table(["case", "shape", "MiB", *names],
                            [[l, s, f"{m:.0f}", *[fmt(t.get(n)) for n in names]] for l, s, m, t in matrix_rows]))
        out.append("\nThroughput in MiB/s of input:\n")
        out.append(md_table(["case", "MiB", *names],
                            [[l, f"{m:.0f}", *[speed(m, t.get(n)) for n in names]] for l, s, m, t in matrix_rows]))
    if string_rows:
        names = list(string_rows[0][3])
        out.append("\n### Read: tables with string columns, native containers\n")
        out.append("Seconds, median. `fastcsv packed` is the parser's own storage (uint64 offsets + UTF-8 arena); "
                   "`fastcsv dict` additionally creates one Python `str` per cell; `fastcsv arrow` wraps the packed "
                   "buffers as `large_string` without copying.\n")
        out.append(md_table(["case", "shape", "MiB", *names],
                            [[l, s, f"{m:.0f}", *[fmt(t.get(n)) for n in names]] for l, s, m, t in string_rows]))
    return "\n".join(out)


def section_write(comp: Competitors, root: Path, keys, scale: float, repeats: int, threads: int):
    rows_out = []
    for key in keys:
        label, base_rows, cols, kind = CASES[key]
        rows = max(1, int(base_rows * scale))
        matrix, columns = make_case(rows, cols, kind)
        data = matrix if matrix is not None else columns
        path = root / f"{key}.csv"
        fc.write_csv(data, path, threads=threads)
        mib = path.stat().st_size / 2**20
        times = {name: timed(fn, repeats) for name, fn in comp.writers(data, matrix, columns, root, threads).items()}
        rows_out.append((label, f"{rows:,} x {cols}", mib, times))
        del matrix, columns, data
        gc.collect()
    names = []
    for _, _, _, t in rows_out:
        for n in t:
            if n not in names:
                names.append(n)
    out = ["### Write\n", "Seconds, median. fastcsv time includes writing the sidecar. Floats are written with "
           "shortest round-trip text by every library except np.savetxt (`%.17g`).\n"]
    out.append(md_table(["case", "shape", "MiB", *names],
                        [[l, s, f"{m:.0f}", *[fmt(t.get(n)) for n in names]] for l, s, m, t in rows_out]))
    return "\n".join(out)


def section_sizes(comp: Competitors, root: Path, scale: float, repeats: int, threads: int):
    out = ["### Read: size scaling (float64, 8 columns)\n",
           "Seconds, median; warm page cache. Same readers as above.\n"]
    rows_out = []
    for base_rows in SIZE_ROWS:
        rows = max(1, int(base_rows * scale))
        matrix, columns, data, path, nosidecar = write_case_files(root, "size", rows, 8, "f64", threads)
        mib = path.stat().st_size / 2**20
        readers = comp.matrix_readers(path, nosidecar, np.float64, threads)
        times = {name: timed(fn, repeats if mib < 200 else max(1, repeats // 3)) for name, fn in readers.items()}
        rows_out.append((f"{rows:,} x 8", mib, times))
        del matrix, data
        gc.collect()
        path.unlink()
        nosidecar.unlink()
    names = list(rows_out[0][2])
    out.append(md_table(["shape", "MiB", *names], [[s, f"{m:.0f}", *[fmt(t.get(n)) for n in names]] for s, m, t in rows_out]))
    return "\n".join(out)


def section_threads(comp: Competitors, root: Path, scale: float, repeats: int, threads_list):
    label, base_rows, cols, kind = CASES["f64_tall"]
    rows = max(1, int(base_rows * scale))
    matrix, columns, data, path, nosidecar = write_case_files(root, "threads", rows, cols, kind, 0)
    mib = path.stat().st_size / 2**20
    out = [f"### Thread scaling ({label}, {rows:,} rows, {mib:.0f} MiB)\n",
           "Seconds, median. The sidecar plan has 4 MiB chunks; the sidecar-free read includes its "
           "parallel inspection pass. Polars shown for reference at the same thread count via "
           "`POLARS_MAX_THREADS` (set at import, so it is only varied when polars is imported fresh).\n"]
    rows_out = []
    for th in threads_list:
        t_read = timed(lambda: fc.read_csv(path, threads=th), repeats)
        t_nosc = timed(lambda: fc.read_csv(nosidecar, threads=th), repeats)
        t_write = timed(lambda: fc.write_csv(data, root / "w_threads.csv", threads=th), repeats)
        t_pl = None
        if comp.pl is not None:
            script = (f"import time,polars as pl; pl.read_csv({str(path)!r}); t0=time.perf_counter(); "
                      f"pl.read_csv({str(path)!r}).to_numpy(); print(time.perf_counter()-t0)")
            env = dict(os.environ, POLARS_MAX_THREADS=str(th))
            samples = []
            for _ in range(max(1, min(repeats, 3))):
                res = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
                if res.returncode == 0:
                    samples.append(float(res.stdout.strip()))
            t_pl = statistics.median(samples) if samples else None
        rows_out.append([th, fmt(t_read), speed(mib, t_read), fmt(t_nosc), fmt(t_write), fmt(t_pl)])
    out.append(md_table(["threads", "read (sidecar)", "MiB/s", "read (no sidecar)", "write", "polars read"], rows_out))
    return "\n".join(out)


MEMORY_SCRIPT = r'''
import resource, sys
path, lib = sys.argv[1], sys.argv[2]
def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
import numpy as np
if lib == "baseline":
    pass
elif lib == "fastcsv numpy":
    import vvtk_fastcsv as fc; x = fc.read_csv(path)
elif lib == "fastcsv torch":
    import vvtk_fastcsv as fc; x = fc.read_csv(path, output="torch")
elif lib == "np.loadtxt":
    x = np.loadtxt(path, delimiter=",", skiprows=1)
elif lib == "pandas (C) .to_numpy()":
    import pandas as pd; x = pd.read_csv(path).to_numpy()
elif lib == "pandas (pyarrow) .to_numpy()":
    import pandas as pd; x = pd.read_csv(path, engine="pyarrow").to_numpy()
elif lib == "pyarrow.csv -> numpy":
    import pyarrow.csv as pacsv; t = pacsv.read_csv(path); x = np.column_stack([c.to_numpy() for c in t.columns])
elif lib == "polars .to_numpy()":
    import polars as pl; x = pl.read_csv(path).to_numpy()
elif lib == "duckdb .fetchnumpy()":
    import duckdb; x = np.column_stack(list(duckdb.sql(f"SELECT * FROM read_csv('{path}')").fetchnumpy().values()))
print(rss())
'''


def section_memory(comp: Competitors, root: Path, scale: float, threads: int):
    label, base_rows, cols, kind = CASES["f64_tall"]
    rows = max(1, int(base_rows * scale))
    matrix, columns, data, path, nosidecar = write_case_files(root, "memory", rows, cols, kind, threads)
    mib = path.stat().st_size / 2**20
    result_mib = matrix.nbytes / 2**20
    libs = ["baseline", "fastcsv numpy", "np.loadtxt"]
    if comp.torch is not None:
        libs.insert(2, "fastcsv torch")
    if comp.pd is not None:
        libs.append("pandas (C) .to_numpy()")
        if comp.pacsv is not None:
            libs.append("pandas (pyarrow) .to_numpy()")
    if comp.pacsv is not None:
        libs.append("pyarrow.csv -> numpy")
    if comp.pl is not None:
        libs.append("polars .to_numpy()")
    if comp.duckdb is not None:
        libs.append("duckdb .fetchnumpy()")
    peaks = {}
    for lib in libs:
        res = subprocess.run([sys.executable, "-c", MEMORY_SCRIPT, str(path), lib], capture_output=True, text=True)
        peaks[lib] = float(res.stdout.strip()) if res.returncode == 0 and res.stdout.strip() else None
    base = peaks.pop("baseline") or 0.0
    out = [f"### Peak memory ({label}, {rows:,} rows, {mib:.0f} MiB CSV, {result_mib:.0f} MiB result)\n",
           "Peak resident set size of a fresh process minus a baseline that only imports NumPy, in MiB. "
           "Includes each library's own import footprint.\n"]
    out.append(md_table(["reader", "peak RSS above baseline (MiB)", "x result size"],
                        [[lib, "-" if p is None else f"{p - base:,.0f}", "-" if p is None else f"{(p - base) / result_mib:.1f}"]
                         for lib, p in peaks.items()]))
    return "\n".join(out)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", default="all", help="comma list of: read,write,sizes,threads,memory (default all)")
    ap.add_argument("--cases", default="all", help="comma list of case keys for read/write sections")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply row counts by this factor")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="vvtk_fastcsv workers; 0 = hardware concurrency")
    ap.add_argument("--thread-list", default=",".join(map(str, THREADS)))
    ap.add_argument("--skip", default="", help="comma list of competitors to skip: pandas,pyarrow,polars,duckdb,torch")
    ap.add_argument("--quick", action="store_true", help="scale 0.05, 2 repeats, no size/memory sections")
    ap.add_argument("--out", help="write the Markdown report to this file as well")
    args = ap.parse_args()
    if args.quick:
        args.scale, args.repeats = 0.05, 2
        if args.section == "all":
            args.section = "read,write,threads"

    sections = ["read", "write", "sizes", "threads", "memory"] if args.section == "all" else args.section.split(",")
    keys = list(CASES) if args.cases == "all" else args.cases.split(",")
    comp = Competitors(set(filter(None, args.skip.split(","))))
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    header = [
        "# vvtk_fastcsv benchmark results\n",
        f"- machine: {cpu}, {os.cpu_count()} logical CPUs, {platform.system()} {platform.release()}",
        f"- python {platform.python_version()}; {comp.versions()}",
        f"- build: {fc.build_info()}",
        f"- threads: {args.threads or os.cpu_count()} (vvtk_fastcsv); competitors use their defaults (all cores)",
        f"- repeats: {args.repeats} (median), warm page cache, scale {args.scale}",
        f"- command: `python benchmarks/benchmark.py {' '.join(sys.argv[1:])}`\n",
    ]
    print("\n".join(header))
    report = list(header)
    with tempfile.TemporaryDirectory(prefix="vvtk-fastcsv-bench-") as tmp:
        root = Path(tmp)
        for section in sections:
            t0 = time.perf_counter()
            if section == "read":
                text = section_read(comp, root, keys, args.scale, args.repeats, args.threads)
            elif section == "write":
                text = section_write(comp, root, keys, args.scale, args.repeats, args.threads)
            elif section == "sizes":
                text = section_sizes(comp, root, args.scale, args.repeats, args.threads)
            elif section == "threads":
                text = section_threads(comp, root, args.scale, args.repeats, [int(x) for x in args.thread_list.split(",")])
            elif section == "memory":
                text = section_memory(comp, root, args.scale, args.threads)
            else:
                raise SystemExit(f"unknown section {section!r}")
            print(text + "\n")
            print(f"({section} section took {time.perf_counter() - t0:.0f}s)\n", file=sys.stderr)
            report.append(text + "\n")
    if args.out:
        Path(args.out).write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"report written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
