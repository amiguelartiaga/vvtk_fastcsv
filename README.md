# vvtk_fastcsv

Fast, strict CSV reading and writing **straight into NumPy arrays and torch tensors**.

The C++20 core memory-maps the file, finds delimiters with runtime-dispatched
AVX2/SSE2 code, converts numbers with [fast_float](https://github.com/fastfloat/fast_float)
and writes every value **directly into the final allocation**. There is no
intermediate column store, no `np.column_stack`, no `torch.from_numpy` copy:
the array or tensor you get back is the memory the parser wrote into.

```python
import vvtk_fastcsv as fc

x = fc.read_numpy("features.csv")                       # (rows, cols) float64 ndarray
t = fc.read_torch("features.csv", dtype="float32")      # parsed directly as float32 tensor
g = fc.read_torch("features.csv", device="cuda")        # pinned host parse + one async copy
fc.write_csv(t, "copy.csv")                             # row-major tensor memory written as-is
```

It is a deliberately **strict** parser for CSV you control or trust: one-byte
delimiter, no quoting, fixed schema. In exchange it is several times faster
than the general-purpose readers on numeric data and uses a fraction of the
memory. See [the contract](#the-csv-contract) and [benchmarks](#benchmarks).

## Contents

- [Install](#install)
- [API at a glance](#api-at-a-glance)
- [Reading](#reading): [matrices](#matrices-numpy--torch), [columns](#columns-dict--pandas--arrow--polars), [foreign files](#files-you-did-not-write), [options](#options)
- [Writing](#writing)
- [What is actually zero-copy](#what-is-actually-zero-copy)
- [The CSV contract](#the-csv-contract)
- [The sidecar](#the-sidecar)
- [How it works](#how-it-works)
- [Benchmarks](#benchmarks)
- [Development](#development)
- [Repository layout](#repository-layout)
- [Limitations and roadmap](#limitations-and-roadmap)

## Install

Requirements: Python 3.10+, a C++20 compiler (GCC 11+, Clang 13+, MSVC 2022).
CMake and Ninja are fetched from PyPI during the build.

```bash
pip install git+https://github.com/amiguelartiaga/vvtk_fastcsv
# optional extras
pip install "vvtk-fastcsv[torch,pandas,arrow,polars] @ git+https://github.com/amiguelartiaga/vvtk_fastcsv"
```

From a checkout:

```bash
pip install -e '.[dev]'          # editable build + test dependencies
pytest -q
```

Check what the build enabled:

```python
>>> fc.build_info()
{'cpp_extension': True, 'fast_float': True, 'simd_scanner': True, 'simd_runtime': 'avx2',
 'direct_buffers': True, 'sidecar_version': 7, 'version': '1.0.0'}
```

## API at a glance

| function | what it does |
|---|---|
| `read_csv(path, schema=None, *, output="numpy", dtype=None, out=None, device=None, pin_memory=False, has_header=None, delimiter=None, threads=0, strings="python", empty_float_is_nan=True, validate_metadata=True, return_stats=False)` | Parse a strict CSV into the requested container. Default: one row-major NumPy matrix. |
| `read_numpy(path, dtype=None, **kw)` | `read_csv(..., output="numpy")`: returns `np.ndarray` of shape `(rows, cols)`. |
| `read_torch(path, dtype=None, device=None, pin_memory=False, **kw)` | `read_csv(..., output="torch")`: returns a contiguous `torch.Tensor`, optionally on a device. |
| `write_csv(data, path, *, schema=None, columns=None, header=True, delimiter=",", metadata=True, threads=0, chunk_rows=0, target_chunk_bytes=4 MiB, float_precision=None)` | Write canonical CSV plus sidecar from a matrix, mapping, DataFrame or Arrow table. Returns the `Path`. |
| `inspect_csv(path, schema, *, has_header=True, delimiter=",", threads=0)` | Row count, per-column string bytes and a chunk plan for a sidecar-free file, computed in parallel. |
| `plan_read(path, schema=None, *, has_header=None, delimiter=None, threads=0)` | The resolved `ReadPlan` (schema, header, delimiter, rows, chunks) that `read_csv` would use. |
| `load_metadata(path)` | The `<path>.fcsv.json` sidecar as a dict. |
| `normalize_schema(schema)` | Canonical `[(name, dtype), ...]` from a mapping or pair list, resolving NumPy/torch dtypes and aliases. |
| `build_info()` | Which accelerations the installed extension has (fast_float, SIMD level, sidecar version). |

Supported column dtypes and what they become:

| schema dtype | aliases accepted | NumPy | torch | notes |
|---|---|---|---|---|
| `int64` | `int`, `i8`, `np.int64`, `torch.int64` | `int64` | `torch.int64` | |
| `int32` | `i4`, `np.int32`, `torch.int32`, narrower ints | `int32` | `torch.int32` | `int8/16`, `uint8/16`, `bool` widen to this |
| `float64` | `float`, `f8`, `double` | `float64` | `torch.float64` | empty cell parses as NaN |
| `float32` | `f4`, `single`, `float16` | `float32` | `torch.float32` | empty cell parses as NaN |
| `string` | `str`, `text`, `object` | object array / packed buffers | list of `str` | packed as `uint64` offsets + UTF-8 arena |
| `skip` | `None`, `drop` | not returned | not returned | field is scanned past, no allocation |

The output containers:

| `output=` | returns | strings | copies |
|---|---|---|---|
| `"numpy"` (default) | 2-D `np.ndarray` | not allowed (use `skip`) | none |
| `"torch"` | 2-D `torch.Tensor` | not allowed (use `skip`) | none on CPU, one H2D copy on CUDA |
| `"dict"` | `{name: 1-D ndarray}` | object arrays of `str` | none for numbers |
| `"torch_dict"` | `{name: 1-D tensor}` | `list[str]` | none for numbers |
| `"packed"` | `{name: ndarray | {"offsets", "data"}}` | raw packed buffers | none |
| `"pandas"` | `pd.DataFrame` | pandas string dtype | pandas may copy strings |
| `"pandas_arrow"` | `pd.DataFrame` with `ArrowDtype` columns | Arrow `large_string` | none |
| `"arrow"` | `pyarrow.Table` | `large_string` | none |
| `"polars"` | `pl.DataFrame` | `str` | none |

Accepted `write_csv` inputs: 2-D or 1-D `np.ndarray`, 2-D or 1-D `torch.Tensor`
(CPU or CUDA; CUDA is copied to host first), `{name: column}` mappings whose
values are arrays, tensors, lists, pandas Series or Arrow arrays,
`pandas.DataFrame`, `polars.DataFrame`, `pyarrow.Table`.

## Reading

`read_csv(path, schema=None, *, output="numpy", ...)` is the single entry
point; `read_numpy` and `read_torch` are thin aliases.

### Matrices (NumPy / torch)

```python
arr = fc.read_csv("m.csv")                               # ndarray, dtype = common numeric dtype
arr = fc.read_csv("m.csv", dtype=np.float32)             # every cell parsed directly as float32
t   = fc.read_csv("m.csv", output="torch")               # torch.Tensor on CPU
t   = fc.read_torch("m.csv", device="cuda", dtype="f4")  # pinned host memory -> one async H2D copy
```

Reuse an allocation (per epoch, inside a `DataLoader`, ...): the parser fills
`out` in place and returns `out[:rows]`.

```python
buf = torch.empty((max_rows, cols), dtype=torch.float32, pin_memory=True)
view = fc.read_csv("m.csv", output="torch", out=buf)
```

Files with an ID or label column: mark it `skip` and the remaining columns
form the matrix.

```python
X = fc.read_numpy("iris.csv", schema={"sl": "f4", "sw": "f4", "pl": "f4", "pw": "f4", "species": "skip"})
```

### Columns (dict / pandas / Arrow / polars)

```python
cols  = fc.read_csv("data.csv", output="dict")          # {name: 1-D ndarray}; strings as object arrays
cols  = fc.read_csv("data.csv", output="torch_dict")    # {name: 1-D tensor}; strings as lists
df    = fc.read_csv("data.csv", output="pandas")
df    = fc.read_csv("data.csv", output="pandas_arrow")  # Arrow-backed strings, no per-row str objects
table = fc.read_csv("data.csv", output="arrow")         # pyarrow.Table (large_string columns)
pl_df = fc.read_csv("data.csv", output="polars")
raw   = fc.read_csv("data.csv", output="packed")        # parser storage: uint64 offsets + UTF-8 arena
```

String columns are parsed into **packed UTF-8**: one `uint64` offsets array
and one contiguous byte arena per column. Arrow and polars wrap those buffers
without copying; `dict`/`pandas` decode them into Python strings in C++.

### Files you did not write

Without a sidecar, `read_csv` infers the schema from the first rows (header
detection, `int64` → `float64` → `string` widening), then runs one parallel
inspection pass to count rows and string bytes per chunk, and finally parses
in parallel into exactly-sized buffers. Pass `schema=` to skip inference and
`has_header=` / `delimiter=` to override detection.

Tolerated deviations from the canonical form: `\r\n` line endings, a missing
final newline, and empty cells in float columns (parsed as NaN, the way pandas
writes missing values). Anything else fails with the row and column:

```
RuntimeError: invalid numeric field at data row 1, column 'b' (float64)
```

### Options

| argument | default | meaning |
|---|---|---|
| `schema` | sidecar / inferred | `{name: dtype}` or `[(name, dtype), ...]`; dtypes `int64 int32 float64 float32 string skip`, NumPy/torch dtypes and aliases (`f4`, `int`, `str`, `None`) accepted |
| `output` | `"numpy"` | `numpy torch dict torch_dict packed pandas pandas_arrow arrow polars` |
| `dtype` | common numeric | matrix element type; every column is parsed directly as it |
| `out` | `None` | preallocated 2-D array / tensor with `>= rows` rows |
| `device`, `pin_memory` | CPU, `False` | torch placement; non-CPU devices get a pinned staging parse + one copy |
| `has_header`, `delimiter` | sidecar / detected | override detection |
| `threads` | `0` | worker threads, `0` = hardware concurrency |
| `empty_float_is_nan` | `True` | empty float cells become NaN instead of errors |
| `strings` | `"python"` | `"packed"` keeps packed buffers in `dict`/`torch_dict` |
| `validate_metadata` | `True` | O(1) stale-sidecar check (size + mtime) |
| `return_stats` | `False` | also return `{rows, chunks, workers, plan, schema, has_header}` |

## Writing

`write_csv(data, path, *, schema=None, columns=None, header=True, delimiter=",",
metadata=True, threads=0, chunk_rows=0, target_chunk_bytes=4 MiB, float_precision=None)`

```python
fc.write_csv(ndarray_2d, "m.csv", columns=["x", "y", "z"])   # strided views, no transpose copy
fc.write_csv(tensor_2d, "t.csv")                              # CPU tensor memory used directly
fc.write_csv({"id": ids, "score": scores, "name": names}, "d.csv")
fc.write_csv(pandas_df, "d.csv", schema={"score": "float32"}) # override inferred dtypes
fc.write_csv(pyarrow_table, "d.csv")                          # Arrow strings written from their buffers
fc.write_csv(polars_df, "d.csv", float_precision=6)
```

The writer formats row blocks on `threads` workers (`std::to_chars`, shortest
round-trip floats by default), writes them in order, and publishes the CSV and
its sidecar atomically (temporary file + rename). A stale sidecar is never
left next to a replaced file.

## What is actually zero-copy

| path | copies |
|---|---|
| `read_csv(output="numpy"/"torch")` | none: parser writes into the returned allocation (or `out`) |
| `read_torch(device="cuda")` | one: pinned host → device, asynchronous |
| `output="dict"` numeric columns | none |
| `output="dict"` string columns | packed arena → Python `str` objects (unavoidable) |
| `output="arrow"/"polars"/"pandas_arrow"` strings | none: Arrow buffers wrap the packed arena |
| `output="pandas"` | pandas may copy on DataFrame construction (string dtype conversion) |
| `write_csv(ndarray/tensor)` | none when dtype matches the schema; one `astype` otherwise |
| `write_csv` Python strings | packed once into a UTF-8 arena before formatting |

## The CSV contract

The fast path assumes:

- UTF-8 / ASCII-compatible text, one-byte delimiter (`,` by default);
- LF line endings (`\r\n` tolerated), optional one-line header;
- fixed schema and column order, no ragged rows;
- **no quoted fields**, no delimiter or newline inside values;
- numeric fields parsable by `from_chars` (no thousands separators, no padding, `nan`/`inf` allowed).

Files written by `write_csv` satisfy this by construction, and any standard
CSV tool can read them. Quoted input is rejected, never silently misparsed.

## The sidecar

`write_csv` stores `<file>.csv.fcsv.json` next to the data so a later read
can allocate exactly and start parallel workers immediately:

```json
{"format":"fastcsv-canonical","version":7,"rows":1000000,"bytes":35123456,
 "mtime_ns":1788451200000000000,"header":true,"delimiter":",",
 "chunk_offsets":[8781010,17564002,26345818],
 "chunk_row_counts":[250431,249992,250107,249470],
 "chunk_byte_counts":[8781000,8782992,8779816,8769638],
 "chunk_string_bytes":[[1380123],[1379987],[1381011],[1379442]],
 "columns":[{"name":"id","dtype":"int64"},{"name":"score","dtype":"float64"},{"name":"name","dtype":"string"}]}
```

Chunks are byte-balanced (4 MiB by default) and always end on a row boundary.
The reader checks file size and mtime before trusting it, and verifies every
offset lands after a newline. `metadata=False` skips the sidecar entirely.

## How it works

```text
read                                               write
----                                               -----
sidecar or parallel inspection -> chunk plan       NumPy / torch / pandas / Arrow columns
            |                                                  |
   allocate final NumPy / torch memory once            strided views (no transpose)
            |                                                  |
  mmap + AVX2/SSE2 delimiter scan per chunk          workers format row blocks (to_chars)
  numbers: from_chars / fast_float, one scan                   |
  strings: memcpy into packed UTF-8 arena            ordered writes + chunk metadata
            |                                                  |
   direct writes into the final buffers              atomic rename of CSV + sidecar
```

The GIL is released for the whole parse and write.

## Benchmarks

Measured on a 13th Gen Intel i7-1360P laptop (16 logical CPUs, 4P+8E), Linux,
Python 3.14, NumPy 2.5, pandas 3.0, PyArrow 25, polars 1.44, DuckDB 1.5,
torch 2.14 (CPU). Warm page cache, tmpfs, median of 3 runs. vvtk_fastcsv uses
all 16 threads; competitors run with their own defaults (also all cores).
Full output with throughput tables: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

### Read: numeric matrices to a NumPy array

Seconds, median. Competitor times include their conversion to a 2-D NumPy array.

| case | shape | MiB | fastcsv numpy | fastcsv numpy (no sidecar) | fastcsv torch | np.loadtxt | pandas (C) .to_numpy() | pandas (pyarrow) .to_numpy() | pyarrow.csv -> numpy | polars .to_numpy() | duckdb .fetchnumpy() |
|---|---|---|---|---|---|---|---|---|---|---|---|
| float64, 2 columns | 2,000,000 x 2 | 74 | 0.010 | 0.013 | 0.010 | 0.620 | 0.277 | 0.027 | 0.033 | 0.015 | 0.080 |
| float64, 8 columns | 1,000,000 x 8 | 147 | 0.020 | 0.028 | 0.019 | 1.159 | 0.536 | 0.049 | 0.087 | 0.032 | 0.193 |
| float64, 128 columns | 50,000 x 128 | 118 | 0.017 | 0.033 | 0.018 | 0.945 | 0.457 | 0.058 | 0.119 | 0.042 | 1.294 |
| float32, 8 columns | 1,000,000 x 8 | 81 | 0.021 | 0.024 | 0.022 | 0.417 | 0.435 | 0.038 | 0.078 | 0.032 | 0.189 |
| int64, 8 columns | 1,000,000 x 8 | 102 | 0.020 | 0.022 | 0.023 | 0.297 | 0.654 | 0.046 | 0.084 | 0.027 | 0.183 |
| int32, 8 columns | 1,000,000 x 8 | 79 | 0.019 | 0.020 | 0.019 | 0.266 | 0.526 | 0.037 | 0.074 | 0.027 | 0.180 |
| int64 / float32 / float64, 9 columns | 1,000,000 x 9 | 115 | 0.024 | 0.027 | 0.024 | 0.786 | 0.564 | 0.060 | 0.096 | 0.038 | 0.209 |

Across these numeric cases vvtk_fastcsv is 1.3 to 2.5x faster than polars,
2.5 to 7x faster than PyArrow and pandas' pyarrow engine, 20 to 30x faster
than pandas' default engine and 15 to 60x faster than `np.loadtxt`, while
handing back the final NumPy array or torch tensor rather than a DataFrame.
The sidecar-free path (one extra parallel inspection pass) stays within 1.5x
of the sidecar path.

### Read: tables with string columns, native containers

Seconds, median. `fastcsv packed` is the parser's own storage (uint64 offsets + UTF-8 arena); `fastcsv dict` additionally creates one Python `str` per cell; `fastcsv arrow` wraps the packed buffers as `large_string` without copying.

| case | shape | MiB | fastcsv packed | fastcsv dict (Python str) | fastcsv arrow | fastcsv pandas | pandas (C) | pandas (pyarrow) | pyarrow.csv | polars | duckdb .arrow() |
|---|---|---|---|---|---|---|---|---|---|---|---|
| int64 / float64 / string, 6 columns | 1,000,000 x 6 | 71 | 0.012 | 0.087 | 0.012 | 0.174 | 0.441 | 0.033 | 0.025 | 0.018 | 0.060 |
| short strings (10 B), 6 columns | 500,000 x 6 | 31 | 0.007 | 0.105 | 0.008 | 0.242 | 0.353 | 0.018 | 0.013 | 0.007 | 0.052 |
| long strings (45 B), 4 columns | 250,000 x 4 | 40 | 0.005 | 0.047 | 0.005 | 0.093 | 0.203 | 0.015 | 0.012 | 0.009 | 0.060 |

### Write

Seconds, median. fastcsv time includes writing the sidecar. Floats are written with shortest round-trip text by every library except np.savetxt (`%.17g`).

| case | shape | MiB | fastcsv | np.savetxt | pandas .to_csv() | pyarrow.csv.write_csv | polars .write_csv() |
|---|---|---|---|---|---|---|---|
| float64, 2 columns | 2,000,000 x 2 | 74 | 0.027 | 2.417 | 3.805 | 0.307 | 0.026 |
| float64, 8 columns | 1,000,000 x 8 | 147 | 0.075 | 2.945 | 7.162 | 0.593 | 0.048 |
| float64, 128 columns | 50,000 x 128 | 118 | 0.067 | 1.794 | 5.684 | 0.462 | 0.055 |
| float32, 8 columns | 1,000,000 x 8 | 81 | 0.063 | 2.903 | 4.405 | 0.538 | 0.031 |
| int64, 8 columns | 1,000,000 x 8 | 102 | 0.045 | 1.525 | 1.515 | 0.222 | 0.029 |
| int32, 8 columns | 1,000,000 x 8 | 79 | 0.040 | 1.515 | 1.296 | 0.204 | 0.023 |
| int64 / float32 / float64, 9 columns | 1,000,000 x 9 | 115 | 0.049 | - | 4.767 | 0.481 | 0.036 |
| int64 / float64 / string, 6 columns | 1,000,000 x 6 | 71 | 0.083 | - | 2.298 | 0.201 | 0.022 |
| short strings (10 B), 6 columns | 500,000 x 6 | 31 | 0.092 | - | 0.413 | 0.021 | 0.012 |
| long strings (45 B), 4 columns | 250,000 x 4 | 40 | 0.042 | - | 0.459 | 0.015 | 0.017 |

Writes are bound by the single ordered write stream on this machine; polars
is about 1.5x faster on numeric data and clearly faster on string-heavy data,
where vvtk_fastcsv also packs the Python strings into UTF-8 first.

### Read: size scaling (float64, 8 columns)

Seconds, median; warm page cache. Same readers as above.

| shape | MiB | fastcsv numpy | fastcsv numpy (no sidecar) | fastcsv torch | np.loadtxt | pandas (C) .to_numpy() | pandas (pyarrow) .to_numpy() | pyarrow.csv -> numpy | polars .to_numpy() | duckdb .fetchnumpy() |
|---|---|---|---|---|---|---|---|---|---|---|
| 100,000 x 8 | 15 | 0.003 | 0.005 | 0.004 | 0.135 | 0.058 | 0.009 | 0.009 | 0.004 | 0.113 |
| 1,000,000 x 8 | 147 | 0.022 | 0.033 | 0.023 | 1.262 | 0.569 | 0.051 | 0.091 | 0.034 | 0.204 |
| 5,000,000 x 8 | 735 | 0.098 | 0.135 | 0.103 | 6.226 | 2.814 | 0.359 | 0.577 | 0.181 | 0.827 |

### Thread scaling (float64, 8 columns, 1,000,000 rows, 147 MiB)

Seconds, median. The sidecar plan has 4 MiB chunks; the sidecar-free read includes its parallel inspection pass.

| threads | read (sidecar) | MiB/s | read (no sidecar) | write |
|---|---|---|---|---|
| 1 | 0.135 | 1,085 | 0.141 | 0.335 |
| 2 | 0.073 | 2,013 | 0.082 | 0.204 |
| 4 | 0.040 | 3,669 | 0.051 | 0.135 |
| 8 | 0.027 | 5,504 | 0.037 | 0.102 |
| 16 | 0.021 | 7,051 | 0.031 | 0.064 |

### Peak memory (float64, 8 columns, 1,000,000 rows, 147 MiB CSV, 61 MiB result)

Peak resident set size (VmHWM) of a fresh process, in MiB: the library's import footprint, and the additional peak while reading. vvtk_fastcsv memory-maps the input, so its read peak includes up to 147 MiB of file pages that are page cache, not allocations; its private allocation is the 61 MiB result.

| reader | import (MiB) | read peak above import (MiB) | x result size |
|---|---|---|---|
| fastcsv numpy | 33 | 208 | 3.4 |
| fastcsv torch | 259 | 211 | 3.5 |
| np.loadtxt | 31 | 64 | 1.1 |
| pandas (C) .to_numpy() | 133 | 129 | 2.1 |
| pandas (pyarrow) .to_numpy() | 133 | 397 | 6.5 |
| pyarrow.csv -> numpy | 78 | 503 | 8.2 |
| polars .to_numpy() | 72 | 318 | 5.2 |
| duckdb .fetchnumpy() | 78 | 415 | 6.8 |

Run your own:

```bash
python benchmarks/benchmark.py --quick                       # ~1 minute smoke run
python benchmarks/benchmark.py --out benchmarks/RESULTS.md   # everything, ~5 minutes
python benchmarks/benchmark.py --section read --cases f64_tall,strings_short --threads 8
python benchmarks/benchmark.py --skip duckdb,torch           # drop competitors
```

## Development

```bash
make install-all     # editable install with torch/pandas/pyarrow/polars/pytest
make test            # pytest
make core-test       # standalone C++ smoke test via CMake/CTest, no Python needed
make bench-smoke     # quick benchmark
```

CMake switches (pass through `CMAKE_ARGS='-D...' pip install .`):

```text
FASTCSV_USE_FAST_FLOAT   ON   fast_float (system package or vendored header)
FASTCSV_ENABLE_SIMD      ON   runtime AVX2/SSE2 scanner dispatch (GCC/Clang, x86)
FASTCSV_ENABLE_IPO       ON   LTO when supported
FASTCSV_NATIVE_ARCH      OFF  -march=native for the non-dispatched code (not for wheels)
FASTCSV_BUILD_PYTHON     ON   build the pybind11 extension
FASTCSV_BUILD_CORE_TESTS OFF  build tests/cpp/core_smoke.cpp
```

GitHub Actions run the C++ and Python test suites on Linux, macOS and Windows,
and `wheels.yml` builds wheels with cibuildwheel on tags.

## Repository layout

```text
cpp/fastcsv/reader.{hpp,cpp}   mmap, SIMD scanning, direct-buffer parser, parallel inspection
cpp/fastcsv/writer.{hpp,cpp}   parallel block formatter, chunk metadata
cpp/bindings.cpp               pybind11 module `_core` (buffer-protocol targets)
src/vvtk_fastcsv/              Python API: _read, _write, _schema, _buffers, _sidecar
third_party/fast_float/        vendored fast_float single header (MIT)
tests/                         pytest suite + tests/cpp/core_smoke.cpp
benchmarks/benchmark.py        comparison with NumPy, pandas, PyArrow, polars
examples/                      numpy_matrix.py, torch_tensor.py, foreign_csv.py, pandas_arrow_polars.py
```

## Limitations and roadmap

- No quoting or escaping, by design. Use pandas/PyArrow for messy CSV.
- Supported dtypes: `int64`, `int32`, `float64`, `float32`, `string` (and `skip`).
  Narrower NumPy/torch types are widened to these.
- Compressed input, NEON scanning on ARM, and column projection without
  scanning skipped fields are natural next steps once profiles ask for them.

## License

MIT. fast_float is vendored under its MIT license (`third_party/fast_float`).
