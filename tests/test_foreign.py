"""Files produced by other tools: no sidecar, CRLF, missing values, headerless, custom delimiters."""
import numpy as np
import pytest

import vvtk_fastcsv as fc


def test_infers_schema_and_header(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("a,b,c\r\n1,2.5,x\r\n2,,y\r\n3,-1e3,zz", encoding="utf-8")
    d, stats = fc.read_csv(p, output="dict", return_stats=True)
    assert stats["schema"] == [("a", "int64"), ("b", "float64"), ("c", "string")]
    assert stats["has_header"] is True
    np.testing.assert_array_equal(d["a"], [1, 2, 3])
    assert d["b"][0] == 2.5 and np.isnan(d["b"][1]) and d["b"][2] == -1000.0
    assert d["c"].tolist() == ["x", "y", "zz"]


def test_headerless_numeric_file(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("1,2\n3,4\n")
    got, stats = fc.read_csv(p, return_stats=True)
    assert stats["has_header"] is False
    np.testing.assert_array_equal(got, [[1, 2], [3, 4]])
    got = fc.read_csv(p, schema=[("x", "float32"), ("y", "float32")])
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, [[1, 2], [3, 4]])
    got = fc.read_csv(p, has_header=True)
    assert got.shape == (1, 2)


def test_semicolon_and_tab_delimiters(tmp_path):
    p = tmp_path / "semi.csv"
    p.write_text("x;y\n1;2\n3;4\n")
    np.testing.assert_array_equal(fc.read_csv(p, delimiter=";"), [[1, 2], [3, 4]])
    fc.write_csv(np.eye(2), p, delimiter="\t")
    assert p.read_text() == "c0\tc1\n1\t0\n0\t1\n"
    np.testing.assert_array_equal(fc.read_csv(p), np.eye(2))


def test_empty_int_cell_is_an_error_but_empty_float_is_nan(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("a,b\n1,\n")
    with pytest.raises(RuntimeError, match="row 0, column 'b'"):
        fc.read_csv(p, schema={"a": "int64", "b": "int64"})
    got = fc.read_csv(p, schema={"a": "int64", "b": "float64"})
    assert np.isnan(got[0, 1])
    with pytest.raises(RuntimeError):
        fc.read_csv(p, schema={"a": "int64", "b": "float64"}, empty_float_is_nan=False)


def test_malformed_rows_are_reported_with_position(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n3,x\n")
    with pytest.raises(RuntimeError, match="row 1, column 'b'"):
        fc.read_csv(p, schema={"a": "int64", "b": "float64"})
    p.write_text("a,b\n1,2\n3\n")
    with pytest.raises(RuntimeError):
        fc.read_csv(p, schema={"a": "int64", "b": "float64"})
    p.write_text("a,b\n1,2,3\n")
    with pytest.raises(RuntimeError):
        fc.read_csv(p, schema={"a": "int64", "b": "float64"})


def test_quoted_fields_are_rejected_not_misparsed(tmp_path):
    p = tmp_path / "q.csv"
    p.write_text('a,b\n"1",2\n')
    with pytest.raises(RuntimeError):
        fc.read_csv(p, schema={"a": "int64", "b": "int64"})


def test_large_sidecar_free_numeric_file_reads_in_parallel(tmp_path, rng):
    m = rng.random((60_000, 4))
    p = tmp_path / "big.csv"
    np.savetxt(p, m, delimiter=",", fmt="%.17g", header="a,b,c,d", comments="")
    got, stats = fc.read_csv(p, threads=4, return_stats=True)
    assert stats["plan"] == "inspect" and stats["chunks"] > 1
    np.testing.assert_array_equal(got, m)


def test_missing_file_and_empty_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        fc.read_csv(tmp_path / "nope.csv")
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        fc.read_csv(p)
    assert fc.read_csv(p, schema={"a": "float64"}).shape == (0, 1)
