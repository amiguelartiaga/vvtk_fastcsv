"""DataFrame round trips. Requires: pip install 'vvtk-fastcsv[pandas,arrow,polars]'."""
import numpy as np
import pandas as pd

import vvtk_fastcsv as fc

n = 200_000
df = pd.DataFrame({
    "id": np.arange(n, dtype=np.int64),
    "score": np.linspace(0, 1, n),
    "label": [f"cls{i % 10}" for i in range(n)],
})
fc.write_csv(df, "frame.csv", threads=0)

print(fc.read_csv("frame.csv", output="pandas").head())
# Arrow-backed strings are built from the parser's packed buffers: no per-row Python objects.
print(fc.read_csv("frame.csv", output="pandas_arrow").dtypes)
print(fc.read_csv("frame.csv", output="arrow").schema)
try:
    print(fc.read_csv("frame.csv", output="polars").head())
except ImportError:
    print("polars not installed")

# The raw parser storage: offsets + one UTF-8 arena per string column.
packed = fc.read_csv("frame.csv", output="packed")
print(packed["label"]["offsets"][:5], bytes(packed["label"]["data"][:12]))
