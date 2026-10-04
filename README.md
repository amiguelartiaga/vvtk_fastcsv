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

Linux, 16-core desktop, tmpfs, Python 3.14, NumPy 2.5, pandas 3.0, PyArrow 25, polars 1.x,
warm page cache, median of 3 runs (`python benchmarks/benchmark.py --scale 0.5 --repeats 3`).
Times in seconds.

**Read** (fastcsv parses into the final NumPy array / torch tensor; the others
build their own containers)

| case | shape | MiB | fastcsv numpy | fastcsv torch | np.loadtxt | pandas | pyarrow | polars |
|---|---|---|---|---|---|---|---|---|
| float64 matrix | 1,000,000 x 4 | 74 | **0.014** | 0.013 | 0.587 | 0.261 | 0.022 | 0.013 |
| float32 matrix | 1,000,000 x 4 | 41 | **0.010** | 0.010 | 0.184 | 0.195 | 0.015 | 0.012 |
| wide float64 | 50,000 x 64 | 59 | **0.009** | 0.010 | 0.447 | 0.220 | 0.021 | 0.013 |
| int64 matrix | 1,000,000 x 6 | 59 | **0.014** | 0.014 | 0.202 | 0.381 | 0.023 | 0.016 |

| case (strings) | shape | MiB | fastcsv packed | fastcsv dict | fastcsv pandas | pandas | pyarrow | polars |
|---|---|---|---|---|---|---|---|---|
| int/float/string | 500,000 x 6 | 35 | **0.007** | 0.036 | 0.079 | 0.207 | 0.016 | 0.009 |
| string heavy | 250,000 x 6 | 16 | **0.005** | 0.048 | 0.109 | 0.160 | 0.007 | 0.005 |

`packed` is the parser's own output (offsets + UTF-8 arena, what Arrow/polars
outputs wrap for free); `dict` is the cost of creating one Python `str` per
cell, which no parser can avoid when you ask for Python strings.

**Write** (fastcsv time includes the sidecar)

| case | MiB | fastcsv | np.savetxt | pandas | polars |
|---|---|---|---|---|---|
| float64 matrix | 74 | 0.028 | 1.702 | 3.370 | 0.020 |
| float32 matrix | 41 | 0.020 | 1.676 | 2.186 | 0.015 |
| wide float64 | 59 | 0.026 | 0.905 | 2.753 | 0.034 |
| int64 matrix | 59 | 0.025 | 1.288 | 1.010 | 0.017 |
| int/float/string | 35 | 0.040 | - | 1.084 | 0.012 |
| string heavy | 16 | 0.047 | - | 0.218 | 0.008 |

Writes are bound by the single ordered write stream on this machine (a raw
74 MiB `write()` takes 18 ms on the same tmpfs); the string cases also pay
for packing Python `str` objects into UTF-8 before formatting. polars is
the one to beat there.

Run your own: `python benchmarks/benchmark.py --scale 0.1 --repeats 3`
(`--no-sidecar` benchmarks the foreign-file path, `--threads N` limits workers).

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
