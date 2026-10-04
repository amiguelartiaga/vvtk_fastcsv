"""Write a NumPy matrix as CSV and read it back straight into a new array."""
import numpy as np

import vvtk_fastcsv as fc

m = np.random.default_rng(0).random((1_000_000, 8), dtype=np.float32)
fc.write_csv(m, "matrix.csv", columns=[f"f{i}" for i in range(8)])  # writes matrix.csv + matrix.csv.fcsv.json

arr, stats = fc.read_csv("matrix.csv", return_stats=True)  # output="numpy" is the default
print(arr.shape, arr.dtype, stats)

# Reuse one allocation across reads (e.g. per epoch): the parser fills `out` in place.
out = np.empty_like(arr)
view = fc.read_csv("matrix.csv", out=out)
assert np.shares_memory(view, out)
