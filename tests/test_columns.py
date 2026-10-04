"""Column outputs: dict, packed, pandas, arrow, polars; strings and sidecar behaviour."""
import json
from pathlib import Path

import numpy as np
import pytest

import vvtk_fastcsv as fc

pd = pytest.importorskip("pandas")


def test_roundtrip_numeric_and_string(tmp_path):
    df = pd.DataFrame({"id": [1, 2, 3], "x": [1.5, 2.5, 3.5], "name": ["a", "b", "c"]})
    p = tmp_path / "x.csv"
    fc.write_csv(df, p, target_chunk_bytes=8)
    got = fc.read_csv(p, output="pandas")
    pd.testing.assert_frame_equal(got, df)


def test_parallel_matches_single_thread_and_stats(tmp_path):
    n = 20_000
    df = pd.DataFrame({
        "id": np.arange(n, dtype=np.int64),
        "x": np.linspace(0, 1, n, dtype=np.float64),
        "s": [f"v{i % 97}" for i in range(n)],
    })
    p = tmp_path / "parallel.csv"
    fc.write_csv(df, p, target_chunk_bytes=4096)
    one = fc.read_csv(p, threads=1, output="pandas")
    many, stats = fc.read_csv(p, threads=4, output="pandas", return_stats=True)
    pd.testing.assert_frame_equal(one, many)
    pd.testing.assert_frame_equal(many, df)
    assert stats["rows"] == n
    assert stats["chunks"] > 1
    assert 1 <= stats["workers"] <= 4


def test_sidecar_adaptive_chunk_metadata(tmp_path):
    n = 5000
    df = pd.DataFrame({"id": np.arange(n, dtype=np.int64), "s": ["x" * (1 + (i % 40)) for i in range(n)]})
    p = tmp_path / "adaptive.csv"
    fc.write_csv(df, p, chunk_rows=0, target_chunk_bytes=2048, threads=3)
    meta = fc.load_metadata(p)
    assert meta["version"] == 7
    assert meta["library_version"] == fc.__version__
    assert meta["rows"] == n
    assert meta["bytes"] == p.stat().st_size
    assert len(meta["chunk_row_counts"]) == len(meta["chunk_byte_counts"]) == len(meta["chunk_string_bytes"])
    assert len(meta["chunk_offsets"]) + 1 == len(meta["chunk_row_counts"])
    assert sum(meta["chunk_row_counts"]) == n
    assert sum(meta["chunk_byte_counts"]) == p.stat().st_size - len(b"id,s\n")
    assert sum(x[0] for x in meta["chunk_string_bytes"]) == sum(len(x) for x in df["s"])
    assert len(meta["chunk_row_counts"]) > 1
    raw = p.read_bytes()
    for off in meta["chunk_offsets"]:
        assert raw[off - 1:off] == b"\n"


def test_packed_string_output_is_offsets_plus_bytes(tmp_path):
    df = pd.DataFrame({"id": np.arange(5, dtype=np.int64), "s": ["a", "bb", "", "ccc", "z"]})
    p = tmp_path / "packed.csv"
    fc.write_csv(df, p, target_chunk_bytes=10)
    packed = fc.read_csv(p, output="packed", threads=4)
    offsets = np.asarray(packed["s"]["offsets"], dtype=np.uint64)
    data = np.asarray(packed["s"]["data"], dtype=np.uint8)
    assert offsets.tolist() == [0, 1, 3, 3, 6, 7]
    assert bytes(data).decode("utf-8") == "abbcccz"
    d = fc.read_csv(p, output="dict", strings="packed")
    assert set(d["s"]) == {"offsets", "data"}


def test_dict_output_and_unicode_strings(tmp_path):
    names = ["ñandú", "日本語", "", "emoji 🙂", "plain"]
    p = tmp_path / "uni.csv"
    fc.write_csv({"i": np.arange(5), "name": names}, p)
    d = fc.read_csv(p, output="dict")
    assert d["name"].dtype == object
    assert d["name"].tolist() == names
    np.testing.assert_array_equal(d["i"], np.arange(5))


def test_sidecar_free_strict_string_file_is_parallel(tmp_path):
    n = 30_000
    df = pd.DataFrame({"id": np.arange(n), "s": [f"alpha{i}" for i in range(n)]})
    p = tmp_path / "no_sidecar.csv"
    fc.write_csv(df, p)
    Path(str(p) + ".fcsv.json").unlink()
    info = fc.inspect_csv(p, {"id": "int64", "s": "string"}, target_chunk_bytes=64 * 1024)
    assert info["rows"] == n
    assert info["string_bytes"] == [sum(len(s) for s in df["s"])]
    assert len(info["chunk_row_counts"]) > 1
    got, stats = fc.read_csv(p, schema={"id": "int64", "s": "string"}, threads=8, output="pandas", return_stats=True)
    pd.testing.assert_frame_equal(got, df)
    assert stats["plan"] == "inspect"
    assert stats["chunks"] > 1


def test_empty_header_only_file(tmp_path):
    df = pd.DataFrame({"a": np.array([], dtype=np.int64), "s": np.array([], dtype=object)})
    p = tmp_path / "empty.csv"
    fc.write_csv(df, p, schema={"a": "int64", "s": "string"})
    got = fc.read_csv(p, output="pandas")
    assert list(got.columns) == ["a", "s"]
    assert len(got) == 0
    packed = fc.read_csv(p, output="packed")
    assert np.asarray(packed["s"]["offsets"]).tolist() == [0]


def test_float_roundtrip_exact(tmp_path):
    vals64 = np.array([0.1, -1234.5, 1e-200, np.pi, np.nextafter(1.0, 2.0), np.inf, -np.inf])
    vals32 = np.array([0.1, -1234.5, 1e-20, 3.402823e20], dtype=np.float32)
    p = tmp_path / "f.csv"
    fc.write_csv({"v64": vals64, "v32": np.resize(vals32, 7)}, p)
    got = fc.read_csv(p, output="dict")
    np.testing.assert_array_equal(got["v64"], vals64)
    assert got["v32"].dtype == np.float32
    np.testing.assert_array_equal(got["v32"], np.resize(vals32, 7))


def test_nan_roundtrip_and_float_precision(tmp_path):
    p = tmp_path / "nan.csv"
    fc.write_csv({"v": np.array([np.nan, 1.0 / 3.0])}, p)
    assert p.read_text() == "v\nnan\n0.3333333333333333\n"
    got = fc.read_csv(p, output="dict")["v"]
    assert np.isnan(got[0]) and got[1] == 1.0 / 3.0
    fc.write_csv({"v": np.array([1.0 / 3.0, 12345.678])}, p, float_precision=4)
    assert p.read_text() == "v\n0.3333\n1.235e+04\n"


def test_reject_delimiter_or_newline_in_string(tmp_path):
    for bad in ["hello,world", "hello\nworld", "hello\rworld"]:
        with pytest.raises(ValueError, match="delimiter or a newline"):
            fc.write_csv({"s": [bad]}, tmp_path / "x.csv")


def test_reject_bad_header_name(tmp_path):
    with pytest.raises(ValueError, match="column name"):
        fc.write_csv({"bad,name": [1]}, tmp_path / "x.csv")


def test_row_limit_chunking(tmp_path):
    p = tmp_path / "row_limit.csv"
    fc.write_csv({"x": np.arange(10, dtype=np.int64)}, p, chunk_rows=3, target_chunk_bytes=10**9)
    assert fc.load_metadata(p)["chunk_row_counts"] == [3, 3, 3, 1]


def test_build_info_shape(build_info):
    assert build_info["cpp_extension"] is True
    assert isinstance(build_info["fast_float"], bool)
    assert build_info["simd_runtime"] in {"avx2", "sse2", "scalar"}
    assert build_info["direct_buffers"] is True
    assert build_info["sidecar_version"] == 7


def test_arrow_and_polars_outputs(tmp_path):
    pa = pytest.importorskip("pyarrow")
    df = pd.DataFrame({"id": [1, 2, 3], "s": ["a", "bb", "ccc"]})
    p = tmp_path / "arrow.csv"
    fc.write_csv(df, p)
    table = fc.read_csv(p, output="arrow")
    assert table.num_rows == 3 and table.column_names == ["id", "s"]
    assert table.column("s").type == pa.large_string()
    got = fc.read_csv(p, output="pandas_arrow")
    assert got["s"].astype(str).tolist() == ["a", "bb", "ccc"]
    pl = pytest.importorskip("polars")
    frame = fc.read_csv(p, output="polars")
    assert frame.shape == (3, 2) and frame["s"].to_list() == ["a", "bb", "ccc"]


def test_arrow_backed_strings_write_zero_copy_path(tmp_path):
    pa = pytest.importorskip("pyarrow")
    values = ["x", "", "yy", "ñ"]
    table = pa.table({"i": pa.array([1, 2, 3, 4], pa.int32()), "s": pa.array(values)})
    p = tmp_path / "pa.csv"
    fc.write_csv(table, p)
    assert p.read_text(encoding="utf-8") == "i,s\n1,x\n2,\n3,yy\n4,ñ\n"
    d = fc.read_csv(p, output="dict")
    assert d["i"].dtype == np.int32 and d["s"].tolist() == values
    big = pa.table({"s": pa.array(values, pa.large_string())})
    fc.write_csv(big, p)
    assert fc.read_csv(p, output="dict")["s"].tolist() == values
    nulls = pa.table({"s": pa.array(["a", None, "c"])})
    fc.write_csv(nulls, p)
    assert fc.read_csv(p, output="dict")["s"].tolist() == ["a", "", "c"]


def test_pandas_missing_strings_become_empty(tmp_path):
    df = pd.DataFrame({"s": ["a", None, "c"]})
    p = tmp_path / "miss.csv"
    fc.write_csv(df, p)
    assert p.read_text() == "s\na\n\nc\n"


def test_sidecar_identity_rejects_stale_metadata(tmp_path):
    p = tmp_path / "stale.csv"
    fc.write_csv({"x": np.arange(10, dtype=np.int64)}, p)
    meta_path = Path(str(p) + ".fcsv.json")
    meta = json.loads(meta_path.read_text())
    meta["mtime_ns"] = int(meta["mtime_ns"]) - 1
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="modification time"):
        fc.read_csv(p)
    np.testing.assert_array_equal(fc.read_csv(p, validate_metadata=False)[:, 0], np.arange(10))
    meta = fc.load_metadata(p)
    meta["bytes"] += 1
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sidecar"):
        fc.read_csv(p)


def test_sidecar_offsets_not_on_row_boundary_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    fc.write_csv({"x": np.arange(2000, dtype=np.int64)}, p, chunk_rows=500)
    meta_path = Path(str(p) + ".fcsv.json")
    meta = json.loads(meta_path.read_text())
    meta["chunk_offsets"][0] += 1
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="row boundary"):
        fc.read_csv(p, validate_metadata=False)


def test_metadata_false_removes_old_sidecar_on_overwrite(tmp_path):
    p = tmp_path / "replace.csv"
    fc.write_csv({"x": [1, 2]}, p)
    meta_path = Path(str(p) + ".fcsv.json")
    assert meta_path.exists()
    fc.write_csv({"x": [3, 4, 5]}, p, metadata=False)
    assert not meta_path.exists()
    np.testing.assert_array_equal(fc.read_csv(p, schema={"x": "int64"})[:, 0], [3, 4, 5])


def test_schema_overrides_sidecar_dtypes(tmp_path):
    p = tmp_path / "ov.csv"
    fc.write_csv({"a": np.arange(4, dtype=np.int64), "b": np.ones(4)}, p)
    d, stats = fc.read_csv(p, schema={"a": "int32", "b": "float32"}, output="dict", return_stats=True)
    assert d["a"].dtype == np.int32 and d["b"].dtype == np.float32
    assert stats["plan"] == "sidecar"
    with pytest.raises(ValueError, match="columns"):
        fc.read_csv(p, schema={"a": "int32"})


def test_write_accepts_dict_of_lists_tensors_and_series(tmp_path):
    p = tmp_path / "dict.csv"
    fc.write_csv({"a": [1, 2, 3], "b": [0.5, 1.5, 2.5], "c": ["x", "y", "z"]}, p)
    assert p.read_text() == "a,b,c\n1,0.5,x\n2,1.5,y\n3,2.5,z\n"
    fc.write_csv({"a": pd.Series([1, 2, 3], dtype="int32"), "s": pd.Series(["p", "q", "r"], dtype="string")}, p)
    d = fc.read_csv(p, output="dict")
    assert d["a"].dtype == np.int32 and d["s"].tolist() == ["p", "q", "r"]
    with pytest.raises(ValueError, match="rows"):
        fc.write_csv({"a": [1, 2, 3], "b": [1, 2]}, p)


def test_schema_dtype_aliases_and_numpy_dtypes():
    assert fc.normalize_schema({"a": np.float32, "b": "int", "c": str, "d": None, "e": "f8"}) == [
        ("a", "float32"), ("b", "int64"), ("c", "string"), ("d", "skip"), ("e", "float64")]
    with pytest.raises(ValueError):
        fc.normalize_schema({"a": "complex128"})
    with pytest.raises(ValueError, match="duplicate"):
        fc.normalize_schema([("a", "int64"), ("a", "int64")])
