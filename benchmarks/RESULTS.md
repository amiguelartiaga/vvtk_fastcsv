# vvtk_fastcsv benchmark results

- machine: 13th Gen Intel(R) Core(TM) i7-1360P, 16 logical CPUs, Linux 7.2.5-3-omarchy
- python 3.14.7; numpy 2.5.3, vvtk_fastcsv 1.0.0, pandas 3.0.6, pyarrow 25.0.1, polars 1.44.2, duckdb 1.5.6, torch 2.14.1+cpu
- build: {'cpp_extension': True, 'fast_float': True, 'simd_scanner': True, 'simd_runtime': 'avx2', 'direct_buffers': True, 'sidecar_version': 7, 'version': '1.0.0'}
- threads: 16 (vvtk_fastcsv); competitors use their defaults (all cores)
- repeats: 3 (median), warm page cache, scale 1.0
- command: `python benchmarks/benchmark.py --repeats 3 --out benchmarks/RESULTS.md`
- note: thread-scaling and peak-memory sections were regenerated after fixing the measurement (same machine, same build)

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

Throughput in MiB/s of input:

| case | MiB | fastcsv numpy | fastcsv numpy (no sidecar) | fastcsv torch | np.loadtxt | pandas (C) .to_numpy() | pandas (pyarrow) .to_numpy() | pyarrow.csv -> numpy | polars .to_numpy() | duckdb .fetchnumpy() |
|---|---|---|---|---|---|---|---|---|---|---|
| float64, 2 columns | 74 | 7,084 | 5,875 | 7,477 | 119 | 265 | 2,733 | 2,205 | 5,001 | 917 |
| float64, 8 columns | 147 | 7,469 | 5,320 | 7,841 | 127 | 274 | 2,993 | 1,681 | 4,545 | 761 |
| float64, 128 columns | 118 | 6,851 | 3,594 | 6,636 | 124 | 257 | 2,029 | 992 | 2,826 | 91 |
| float32, 8 columns | 81 | 3,859 | 3,425 | 3,678 | 195 | 186 | 2,125 | 1,039 | 2,513 | 429 |
| int64, 8 columns | 102 | 5,020 | 4,635 | 4,420 | 344 | 156 | 2,240 | 1,211 | 3,845 | 559 |
| int32, 8 columns | 79 | 4,234 | 3,951 | 4,205 | 298 | 151 | 2,152 | 1,066 | 2,970 | 440 |
| int64 / float32 / float64, 9 columns | 115 | 4,773 | 4,263 | 4,784 | 147 | 205 | 1,929 | 1,205 | 3,024 | 552 |

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
