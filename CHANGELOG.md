# Changelog

## 1.0.0

First release of `vvtk_fastcsv`, grown from the `fastcsv_v1.0` sketch.

- Parse straight into caller-owned memory: NumPy arrays and torch tensors are
  allocated once and filled in place (strided column views over row-major
  matrices; `out=` to reuse an allocation; `device=` for one pinned-host to
  GPU copy).
- Numeric fields are converted in a single scan with `from_chars`/fast_float
  (no separate delimiter search); fast_float is vendored.
- Parallel, byte-balanced inspection pass for files without a sidecar, so
  foreign CSVs also parse in parallel.
- Parallel writer: workers format row blocks, the main thread writes them in
  order and records sidecar chunk metadata. Matrices are written directly from
  row-major memory.
- Schema inference (header detection, int64/float64/string widening), `skip`
  columns, `int32`, CRLF tolerance, empty float cells as NaN, column/row
  positions in error messages.
- Outputs: `numpy`, `torch`, `dict`, `torch_dict`, `packed`, `pandas`,
  `pandas_arrow`, `arrow`, `polars`. Inputs for writing: matrices, mappings,
  pandas/polars frames, Arrow tables (Arrow-backed strings are written from
  their buffers).
- Sidecar format version 7 (reads version 6 files).
