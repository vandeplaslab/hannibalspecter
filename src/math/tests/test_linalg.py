import numpy
import pytest
import scipy.sparse
import scipy.sparse.linalg

from ..core import *
from ..core import _numpy_to_sparse, _sparse_to_numpy
from ..linalg import *
from ..linalg import (
    _arpack,
    _convert_datatype,
    _fbpca,
    _lobpcg,
    _numpy_gesdd,
    _order,
    _propack,
    _pytorch,
    _pytorch_randomized,
    _randomized,
    _recycling_randomized,
    _scipy_gesdd,
    _scipy_gesvd,
    _sparse_arpack,
    _sparse_fbpca,
    _sparse_lobpcg,
    _sparse_propack,
    _sparse_randomized,
    _svd_arraytype,
    _svd_invert_arraytype,
    _svd_invert_transpose,
    _svd_transpose,
    _svdecon,
    _unpack,
)


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_qr(a, b):
    c = numpy.random.rand(a, b)
    q, r = qr(c)

    assert q.shape[0] == c.shape[0]
    assert q.shape[1] == c.shape[0]
    assert r.shape == c.shape
    assert numpy.linalg.norm(q @ q.T - numpy.eye(q.shape[0]), 1) < 1e-8
    assert numpy.linalg.norm(q @ r - c, 1) < 1e-8


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_lu(a, b):
    c = numpy.random.rand(a, b)
    p, l, u = lu(c)

    assert p.shape[0] == c.shape[0]
    assert p.shape[1] == c.shape[0]
    assert l.shape[0] == c.shape[0]
    if a > b:
        assert l.shape[1] == c.shape[1]
        assert u.shape[0] == c.shape[1]
    else:
        assert l.shape[1] == c.shape[0]
        assert u.shape[0] == c.shape[0]
    assert u.shape[1] == c.shape[1]
    assert numpy.linalg.norm(p @ l @ u - c, 1) < 1e-8


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_eigh(a, b):
    c = numpy.random.rand(a, b)
    u, v = eigh(c.T @ c)

    assert u.shape[0] == c.shape[1]
    assert v.shape == (c.shape[1], c.shape[1])
    assert numpy.linalg.norm(v @ v.T - numpy.eye(v.shape[0]), 1) < 1e-8
    assert numpy.linalg.norm((v * u) @ v.T - c.T @ c, 1) < 1e-8


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_eigvalsh(a, b):
    c = numpy.random.rand(a, b)
    s = eigvalsh(c.T @ c)

    assert s.shape[0] == b
    assert abs(numpy.linalg.norm(s, 2) - numpy.linalg.norm(c.T @ c, "fro")) < 1e-4


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("atype", ["numpy", "scipy.sparse"])
@pytest.mark.parametrize(
    "dtype, dtype_check",
    [
        ("int8", "float32"),
        ("int16", "float32"),
        ("int32", "float32"),
        ("float32", "float32"),
        ("float64", "float64"),
    ],
)
def test_norm(a, b, atype, dtype, dtype_check):
    c = zeros((a, b), atype, dtype)
    d = norm(c)

    assert d.dtype == dtype_check
    assert d == 0


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("ord", [1, 2, "fro"])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "float32", "float64"])
def test_norm(a, b, dtype, ord):
    c = astype(numpy.random.rand(a, b), dtype)
    d = astype(scipy.sparse.csc_matrix(c), dtype)
    c_norm = norm(c, ord)
    d_norm = norm(d, ord)

    assert c.dtype == d.dtype
    assert abs(c_norm - d_norm) < 1e-4


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("ord", [1, 2, "fro"])
@pytest.mark.parametrize("keepdims", [True, False])
def test_norm_keepdims(a, b, ord, keepdims):
    c = numpy.random.rand(a, b)

    assert (
        norm(c, ord, keepdims=keepdims).shape
        == numpy.linalg.norm(c, ord, keepdims=keepdims).shape
    )


@pytest.mark.filterwarnings("ignore:The type of input")
def test__svd_arraytype():
    a = randn(10, 5, atype="numpy", dtype="float32")
    method = "scipy_gesvd"
    hs_math, a_out = _svd_arraytype(a, method)

    b = scipy.sparse.csc_matrix(a)
    method = "scipy_gesvd"
    hs_math_b, b_out = _svd_arraytype(b, method)

    assert hs_math == "numpy"
    assert (a_out == a).all() == True
    assert hs_math_b == "scipy.sparse"
    assert (b_out == b).all() == True


@pytest.mark.filterwarnings("ignore:The type of input")
def test__svd_invert_arraytype():
    a = randn(10, 5, atype="numpy", dtype="float32")
    b = scipy.sparse.csc_matrix(a)
    method = "scipy_gesvd"
    hs_math, b_out = _svd_arraytype(b, method)
    b_invert = _svd_invert_arraytype(b_out, hs_math)

    z = (b, b, b)
    z_invert = _svd_invert_arraytype(z, hs_math)

    assert find_package(b_invert)[1] == hs_math
    assert (b.toarray() == b_invert.toarray()).all() == True
    assert find_package(z_invert[0])[1] == hs_math
    assert isinstance(z_invert, tuple) == True


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test__svd_transpose(a, b):
    c = numpy.random.rand(a, b)
    check = False if c.shape[0] >= c.shape[1] else True
    trans_arg, d = _svd_transpose(c)

    assert trans_arg == check
    if trans_arg is True:
        assert d.shape[0] == c.shape[1]
        assert d.shape[1] == c.shape[0]
        assert (transpose(d) == c).all() == True
    else:
        assert d.shape == c.shape
        assert (d == c).all() == True


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__svd_invert_transpose(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    q_true = numpy.linalg.svd(c, full_matrices=False, compute_uv=compute_uv)
    trans_arg, d = _svd_transpose(c)
    q = numpy.linalg.svd(d, full_matrices=False, compute_uv=compute_uv)
    l = _svd_invert_transpose(q, trans_arg)

    assert isinstance(l, type(q_true)) == True
    if compute_uv is True:
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(l[0]), q_true[0])) - numpy.eye(l[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert numpy.linalg.norm(q_true[1] - l[1], 2) < 1e-4
        assert (
            numpy.linalg.norm(
                absolute(matmul(l[2], transpose(q_true[2]))) - numpy.eye(l[2].shape[0]),
                2,
            )
            < 1e-4
        )
    else:
        assert numpy.linalg.norm(q_true - l, 2) < 1e-4


@pytest.mark.parametrize("compute_uv", [True, False])
def test__unpack(compute_uv):
    a = numpy.random.rand(10, 5)
    a1 = (a, a, a)
    a2 = [a, a, a]
    a3 = [a]
    a4 = a

    b = _unpack(a, compute_uv)
    b1 = _unpack(a1, compute_uv)
    b2 = _unpack(a2, compute_uv)
    b3 = _unpack(a3, compute_uv)
    b4 = _unpack(a4, compute_uv)

    if compute_uv is True:
        assert (a == b).all() == True
        assert len(a1) == len(b1)
        assert len(a2) == len(b2)
        assert isinstance(b2, tuple)
        assert (a3 == b3).all() == True
        assert (a4 == b4).all() == True
    else:
        assert isinstance(b, numpy.ndarray)
        assert isinstance(b1, numpy.ndarray)
        assert isinstance(b2, numpy.ndarray)
        assert isinstance(b3, numpy.ndarray)
        assert isinstance(b4, numpy.ndarray)


def test__order():
    a = numpy.random.rand(
        100,
    )
    b = numpy.random.rand(20, 20)
    q = numpy.linalg.svd(b, compute_uv=True, full_matrices=False)
    q_copy = list(q)
    q_copy[0], q_copy[1], q_copy[2] = (
        q_copy[0][:, ::-1],
        q_copy[1][::-1],
        q_copy[2][::-1, :],
    )
    q_ordered = _order(tuple(q_copy), 0)
    q_ordered2 = _order(tuple(q_copy), 10)
    u = numpy.linalg.svd(b, compute_uv=False, full_matrices=False)

    assert (_order(a, 0) == a[argsort(a)[::-1]]).all() == True
    assert _order(a, 10).shape == a[:10].shape
    assert (q_ordered[0] == q[0]).all() == True
    assert (q_ordered[1] == q[1]).all() == True
    assert (q_ordered[2] == q[2]).all() == True
    assert q_ordered2[0].shape[1] == 10
    assert q_ordered2[1].shape[0] == 10
    assert q_ordered2[2].shape[0] == 10
    assert isinstance(_order(u, 0), numpy.ndarray) == True


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "float32", "float64"])
def test__convert_datatype(dtype):
    b = numpy.random.rand(20, 20)
    q = numpy.linalg.svd(b, compute_uv=True, full_matrices=False)
    b_out = _convert_datatype(b, dtype)
    q_out = _convert_datatype(q, dtype)

    assert b_out.dtype == dtype
    assert q_out[0].dtype == dtype
    assert q_out[1].dtype == dtype
    assert q_out[2].dtype == dtype


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__numpy_gesdd(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    d = _numpy_gesdd(c, compute_uv)

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == min(a, b)
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, "fro")) < 1e-4
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == min(a, b)
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, "fro")) < 1e-4


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__scipy_gesdd(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    d = _scipy_gesdd(c, compute_uv)

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == min(a, b)
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, "fro")) < 1e-4
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == min(a, b)
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, "fro")) < 1e-4


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__scipy_gesvd(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    d = _scipy_gesvd(c, compute_uv)

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == min(a, b)
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, "fro")) < 1e-4
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == min(a, b)
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, "fro")) < 1e-4


@pytest.mark.parametrize("a", [20, 50, 100])
@pytest.mark.parametrize("b", [20, 50, 150])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__randomized(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    d = _randomized(c, 1, compute_uv, 10, 2, None, "auto")

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == 1
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, 2)) < 1e-2
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == 1
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, 2)) < 1e-2


@pytest.mark.parametrize("a", [20, 50, 100])
@pytest.mark.parametrize("b", [20, 50, 150])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__arpack(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    d = _arpack(c, 1, compute_uv, None, 2, None, None)

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == 1
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, 2)) < 1e-2
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == 1
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, 2)) < 1e-2


@pytest.mark.filterwarnings("ignore:Exited at iteration")
@pytest.mark.parametrize("a", [20, 50, 100])
@pytest.mark.parametrize("b", [20, 50, 150])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__lobpcg(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    d = _lobpcg(c, 1, compute_uv, None, 2, None, None)

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == 1
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, 2)) < 1e-2
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == 1
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, 2)) < 1e-2


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("a", [20, 50, 100])
@pytest.mark.parametrize("b", [20, 50, 150])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__propack(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    d = _propack(c, 1, compute_uv, None, 2, None, None)
    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == 1
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, 2)) < 1e-2
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == 1
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, 2)) < 1e-2


@pytest.mark.parametrize("a", [100, 110, 120])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__svdecon(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    d = _svdecon(c, compute_uv)

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == min(a, b)
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, "fro")) < 1e-4
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == min(a, b)
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, "fro")) < 1e-4


@pytest.mark.parametrize("a", [20, 50, 100])
@pytest.mark.parametrize("b", [20, 50, 150])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__fbpca(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    d = _fbpca(c, 1, compute_uv)

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == 1
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, 2)) < 1e-2
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == 1
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, 2)) < 1e-2


@pytest.mark.parametrize("a", [20, 50, 100])
@pytest.mark.parametrize("b", [20, 50, 150])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__pytorch_randomized(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    d = _pytorch_randomized(c, compute_uv, 1, 2)

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == 1
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, 2)) < 2e-1
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == 1
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, 2)) < 1e-1


@pytest.mark.parametrize("a", [20, 50, 100])
@pytest.mark.parametrize("b", [20, 50, 150])
@pytest.mark.parametrize("v0", [None, True])
@pytest.mark.parametrize("compute_uv", [True, False])
@pytest.mark.parametrize("recycling", [0, 1, 2])
@pytest.mark.parametrize("iter_type", ["qr", "lu", "power"])
def test__recycling_randomized(a, b, v0, compute_uv, recycling, iter_type):
    c = numpy.random.rand(a, b)
    v0 = (
        (numpy.linalg.svd(c, compute_uv=True, full_matrices=False)[2])[0, :]
        if v0 is True
        else v0
    )
    d = _recycling_randomized(c, compute_uv, v0, 1, 10, 2, recycling, None, iter_type)

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == 1
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, 2)) < 1e-2
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == 1
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, 2)) < 1e-2


@pytest.mark.parametrize("a", [20, 50, 100])
@pytest.mark.parametrize("b", [20, 50, 150])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__sparse_arpack(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    c_sparse = scipy.sparse.csc_matrix(c)
    d = _sparse_arpack(c_sparse, 1, compute_uv, None, 2, None, None)

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == 1
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, 2)) < 1e-2
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == 1
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, 2)) < 1e-2


@pytest.mark.filterwarnings("ignore:Exited at iteration")
@pytest.mark.parametrize("a", [20, 50, 100])
@pytest.mark.parametrize("b", [20, 50, 150])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__sparse_lobpcg(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    c_sparse = scipy.sparse.csc_matrix(c)
    d = _sparse_lobpcg(c_sparse, 1, compute_uv, None, 2, None, None)

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == 1
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, 2)) < 1e-2
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == 1
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, 2)) < 1e-2


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("a", [20, 50, 100])
@pytest.mark.parametrize("b", [20, 50, 150])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__sparse_propack(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    c_sparse = scipy.sparse.csc_matrix(c)
    d = _sparse_propack(c_sparse, 1, compute_uv, None, 2, None, None)
    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == 1
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, 2)) < 1e-2
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == 1
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, 2)) < 1e-2


@pytest.mark.parametrize("a", [20, 50, 100])
@pytest.mark.parametrize("b", [20, 50, 150])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__sparse_fbpca(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    c_sparse = scipy.sparse.csc_matrix(c)
    d = _sparse_fbpca(c_sparse, 1, compute_uv)

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == 1
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, 2)) < 1e-2
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == 1
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, 2)) < 1e-2


@pytest.mark.parametrize("a", [20, 50, 100])
@pytest.mark.parametrize("b", [20, 50, 150])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__sparse_randomized(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    c_sparse = scipy.sparse.csc_matrix(c)
    d = _sparse_randomized(c, 1, compute_uv, 10, 2, None, "auto")

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == 1
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, 2)) < 1e-2
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == 1
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, 2)) < 1e-2


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("compute_uv", [True, False])
def test__pytorch(a, b, compute_uv):
    c = numpy.random.rand(a, b)
    d = _pytorch(c, compute_uv)

    if compute_uv is True:
        assert isinstance(d, tuple) == True
        assert (
            numpy.linalg.norm(
                absolute(matmul(transpose(d[0]), d[0])) - numpy.eye(d[0].shape[1]),
                2,
            )
            < 1e-4
        )
        assert d[1].size == min(a, b)
        assert (
            numpy.linalg.norm(
                absolute(matmul(d[2], transpose(d[2]))) - numpy.eye(d[2].shape[0]),
                2,
            )
            < 1e-4
        )
        assert abs(numpy.linalg.norm(d[1], 2) - numpy.linalg.norm(c, "fro")) < 1e-4
    else:
        assert isinstance(d, numpy.ndarray) == True
        assert d.size == min(a, b)
        assert abs(numpy.linalg.norm(d, 2) - numpy.linalg.norm(c, "fro")) < 1e-4


@pytest.mark.filterwarnings("ignore:The type of input")
@pytest.mark.parametrize("a", [numpy.random.rand(50, 40)])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("compute_uv", [True, False])
@pytest.mark.parametrize(
    "method",
    [
        "numpy_gesdd",
        "scipy_gesdd",
        "scipy_gesvd",
        "randomized",
        "arpack",
        "lobpcg",
        "svdecon",
        "fbpca",
        "recycling_randomized",
        "pytorch",
        "pytorch_randomized",
        "sparse_arpack",
        "sparse_lobpcg",
        "sparse_fbpca",
        "sparse_randomized",
    ],
)
def test_svd(a, sparse, compute_uv, method):
    a = scipy.sparse.csc_matrix(a) if sparse is True else a
    q = svd(a, {"method": method, "compute_uv": compute_uv})
    assert (
        isinstance(q, tuple)
        or isinstance(q, numpy.ndarray)
        or isinstance(q, scipy.sparse.spmatrix)
    ) == True


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "float32", "float64"])
def test_pinv(a, b, dtype):
    c = astype(numpy.random.rand(a, b), dtype)

    assert (pinv(c) == numpy.linalg.pinv(c)).all() == True
