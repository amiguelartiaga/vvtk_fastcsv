"""Direct NumPy / torch matrix paths."""
from pathlib import Path

import numpy as np
import pytest

import vvtk_fastcsv as fc

torch = pytest.importorskip("torch", reason="torch tests need torch installed") if False else None
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")


def test_numpy_matrix_roundtrip_float64(tmp_path, rng):
    m = rng.random((20_000, 5))
    p = tmp_path / "m.csv"
    fc.write_csv(m, p, target_chunk_bytes=64 * 1024)
    got, stats = fc.read_csv(p, return_stats=True)
    assert got.shape == m.shape and got.dtype == np.float64
    np.testing.assert_array_equal(got, m)
    assert stats["plan"] == "sidecar"
    assert stats["chunks"] > 1
    assert got.flags.c_contiguous


def test_numpy_matrix_int_and_float32_exact(tmp_path, rng):
    ints = rng.integers(-(2**62), 2**62, size=(1000, 3), dtype=np.int64)
    p = tmp_path / "i.csv"
    fc.write_csv(ints, p)
    np.testing.assert_array_equal(fc.read_numpy(p), ints)
    assert fc.read_numpy(p).dtype == np.int64

    f32 = rng.random((1000, 3), dtype=np.float32)
    f32[0, 0] = np.float32(3.402823e38)
    f32[0, 1] = np.float32(1e-20)
    q = tmp_path / "f.csv"
    fc.write_csv(f32, q)
    got = fc.read_numpy(q)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, f32)


def test_dtype_override_parses_directly(tmp_path):
    m = np.arange(12, dtype=np.int64).reshape(4, 3)
    p = tmp_path / "o.csv"
    fc.write_csv(m, p)
    got = fc.read_numpy(p, dtype=np.float32)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, m.astype(np.float32))
    got = fc.read_csv(p, dtype="int32")
    assert got.dtype == np.int32


def test_mixed_numeric_schema_uses_common_dtype(tmp_path):
    data = {"a": np.array([1, 2, 3], dtype=np.int64), "b": np.array([1.25, 2.5, 3.75], dtype=np.float32)}
    p = tmp_path / "mixed.csv"
    fc.write_csv(data, p)
    arr = fc.read_csv(p, output="numpy")
    assert arr.dtype == np.float64
    np.testing.assert_allclose(arr, np.column_stack([data["a"], data["b"]]))


def test_out_parameter_numpy_reuses_memory(tmp_path):
    m = np.arange(30, dtype=np.float64).reshape(10, 3)
    p = tmp_path / "out.csv"
    fc.write_csv(m, p)
    out = np.zeros((16, 3))
    view = fc.read_csv(p, out=out)
    assert view.shape == (10, 3)
    assert np.shares_memory(view, out)
    np.testing.assert_array_equal(out[:10], m)
    with pytest.raises(ValueError, match="dtype"):
        fc.read_csv(p, out=np.zeros((10, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        fc.read_csv(p, out=np.zeros((5, 3)))
    with pytest.raises(ValueError, match="contiguous"):
        fc.read_csv(p, out=np.zeros((3, 10)).T)


def test_skip_columns_give_numeric_matrix(tmp_path):
    data = {"id": np.arange(6), "label": ["a", "b", "c", "d", "e", "f"], "x": np.linspace(0, 1, 6)}
    p = tmp_path / "skip.csv"
    fc.write_csv(data, p)
    with pytest.raises(ValueError, match="label"):
        fc.read_csv(p, output="numpy")
    got, stats = fc.read_csv(p, schema={"id": "int64", "label": "skip", "x": "float64"}, return_stats=True)
    assert stats["plan"] == "sidecar"  # skipping a sidecar string column still reuses the plan
    np.testing.assert_array_equal(got, np.column_stack([data["id"], data["x"]]))
    got = fc.read_csv(p, schema={"id": None, "label": "skip", "x": "float32"})
    assert got.shape == (6, 1) and got.dtype == np.float32


def test_wide_matrix_parallel(tmp_path, rng):
    m = rng.random((2000, 64))
    p = tmp_path / "wide.csv"
    fc.write_csv(m, p, target_chunk_bytes=128 * 1024, threads=4)
    got, stats = fc.read_csv(p, threads=4, return_stats=True)
    assert stats["chunks"] > 1 and 1 <= stats["workers"] <= 4
    np.testing.assert_array_equal(got, m)


def test_single_column_vector_input(tmp_path):
    v = np.array([1.0, 2.0, 3.0])
    p = tmp_path / "v.csv"
    fc.write_csv(v, p, columns=["v"])
    assert Path(p).read_text() == "v\n1\n2\n3\n"
    assert fc.read_numpy(p).shape == (3, 1)


def test_empty_matrix(tmp_path):
    p = tmp_path / "empty.csv"
    fc.write_csv(np.empty((0, 3)), p)
    got = fc.read_numpy(p)
    assert got.shape == (0, 3)


@needs_torch
def test_torch_matrix_roundtrip_and_direct_parse(tmp_path, rng):
    m = rng.random((5000, 4), dtype=np.float32)
    p = tmp_path / "t.csv"
    fc.write_csv(torch.from_numpy(m), p, columns=list("abcd"), target_chunk_bytes=32 * 1024)
    t, stats = fc.read_torch(p, threads=4, return_stats=True)
    assert isinstance(t, torch.Tensor) and t.dtype == torch.float32 and t.shape == (5000, 4)
    assert t.is_contiguous()
    assert torch.equal(t, torch.from_numpy(m))
    assert stats["chunks"] > 1

    out = torch.empty((8000, 4), dtype=torch.float32)
    view = fc.read_csv(p, output="torch", out=out)
    assert view.data_ptr() == out.data_ptr()
    assert torch.equal(out[:5000], torch.from_numpy(m))

    t64 = fc.read_torch(p, dtype=torch.float64)
    assert t64.dtype == torch.float64
    assert torch.allclose(t64, torch.from_numpy(m).double())


@needs_torch
def test_torch_int_matrix_and_torch_dict(tmp_path):
    t = torch.arange(24, dtype=torch.int32).reshape(6, 4)
    p = tmp_path / "ti.csv"
    fc.write_csv(t, p)
    got = fc.read_torch(p)
    assert got.dtype == torch.int32 and torch.equal(got, t)
    d = fc.read_csv(p, output="torch_dict")
    assert list(d) == ["c0", "c1", "c2", "c3"]
    assert torch.equal(d["c1"], t[:, 1])


@needs_torch
def test_torch_device_argument_cpu_and_unavailable_cuda(tmp_path):
    p = tmp_path / "dev.csv"
    fc.write_csv(np.eye(3), p)
    t = fc.read_torch(p, device="cpu")
    assert t.device.type == "cpu"
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        g = fc.read_torch(p, device="cuda")
        assert g.device.type == "cuda"
        assert torch.equal(g.cpu(), torch.eye(3, dtype=torch.float64))


@needs_torch
def test_torch_out_dtype_mismatch_rejected(tmp_path):
    p = tmp_path / "mm.csv"
    fc.write_csv(np.eye(2), p)
    with pytest.raises(ValueError, match="dtype"):
        fc.read_csv(p, output="torch", out=torch.empty((2, 2), dtype=torch.float32))
    with pytest.raises(TypeError):
        fc.read_csv(p, output="torch", out=np.empty((2, 2)))
