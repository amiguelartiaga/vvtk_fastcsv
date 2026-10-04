"""Read a CSV that was not produced by vvtk_fastcsv: schema inference, CRLF, blanks, labels."""
from pathlib import Path

import vvtk_fastcsv as fc

Path("iris_like.csv").write_text(
    "sepal_length,sepal_width,petal_length,species\r\n"
    "5.1,3.5,1.4,setosa\r\n"
    "4.9,,1.4,setosa\r\n"
    "6.3,3.3,6.0,virginica\r\n",
    encoding="utf-8",
)

# Everything inferred: header, int/float/string columns. Strings come back as object arrays.
columns, stats = fc.read_csv("iris_like.csv", output="dict", return_stats=True)
print(stats["schema"], stats["has_header"])
print(columns)

# Feature matrix only: skip the label column, parse directly as float32.
features = fc.read_csv("iris_like.csv", schema={"sepal_length": "f4", "sepal_width": "f4",
                                                "petal_length": "f4", "species": "skip"})
print(features)

# Or tell it the dtype for all kept columns at once:
import numpy as np  # noqa: E402

features = fc.read_numpy("iris_like.csv", dtype=np.float32,
                         schema=[("sepal_length", "float64"), ("sepal_width", "float64"),
                                 ("petal_length", "float64"), ("species", "skip")])
print(features.dtype, features.shape)
