import numpy
import pytest
import scipy.sparse
import scipy.sparse.linalg

from ..core import *
from ..core import _numpy_to_sparse, _sparse_to_numpy


@pytest.mark.parametrize(
    "x, package_check",
    [
        (numpy.random.rand(5, 5), "numpy"),
        (scipy.sparse.csc_matrix((5, 5)), "scipy.sparse"),
    ],
)
def test_find_package(x, package_check):
    x, package = find_package(x)
    assert package_check == package


def test_sqrt():
    a = numpy.random.rand(5, 5)
    b = scipy.sparse.csc_matrix(a)

    assert (sqrt(a) == sqrt(b)).all() == True
    assert (sqrt(a)).shape == a.shape
    assert (sqrt(b)).shape == b.shape


def test_sqrt_write():
    a = numpy.random.rand(5, 5)
    b = scipy.sparse.csc_matrix(a)
    id_a_check, id_b_check, id_b_data_check = id(a), id(b), id(b.data)
    a = sqrt(a)
    b = sqrt(b)

    assert id_a_check != id(a)
    assert id_b_check == id(b)
    assert id_b_data_check != id(b.data)


def test_sqrt_overwrite():
    a = numpy.random.rand(5, 5)
    b = scipy.sparse.csc_matrix(a)
    id_a_check, id_b_check, id_b_data_check = id(a), id(b), id(b.data)
    a = sqrt(a, out=a)
    b = sqrt(b, out=b.data)

    assert id_a_check == id(a)
    assert id_b_check == id(b)
    assert id_b_data_check == id(b.data)


def test_clip_write():
    a = numpy.random.rand(5, 5)
    b = scipy.sparse.csc_matrix(a)
    id_a_check, id_b_check, id_b_data_check = id(a), id(b), id(b.data)
    a = clip(a, -0.5, 0.5)
    b = clip(b, -0.5, 0.5)

    assert id_a_check != id(a)
    assert id_b_check == id(b)
    assert id_b_data_check != id(b.data)


def test_clip_overwrite():
    a = numpy.random.rand(5, 5)
    b = scipy.sparse.csc_matrix(a)
    id_a_check, id_b_check, id_b_data_check = id(a), id(b), id(b.data)
    a = clip(a, -0.5, 0.5, out=a)
    b = clip(b, -0.5, 0.5, out=b.data)

    assert id_a_check == id(a)
    assert id_b_check == id(b)
    assert id_b_data_check == id(b.data)


@pytest.mark.parametrize(
    "x",
    [
        numpy.ones((5, 1)),
        numpy.zeros((5, 1)),
        numpy.random.randint(0, 1, (100, 1)),
        scipy.sparse.csc_matrix(numpy.ones((5, 1))),
        scipy.sparse.csc_matrix(numpy.zeros((5, 1))),
        scipy.sparse.csc_matrix(numpy.random.randint(0, 1, (100, 1))),
    ],
)
def test_count_nonzero(x):

    assert count_nonzero(x) == x.sum()


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("atype", ["numpy", "scipy.sparse"])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "float32", "float64"])
def test_zeros(a, b, atype, dtype):
    c = zeros((a, b), atype=atype, dtype=dtype)

    assert c.shape == (a, b)
    assert find_package(c)[1] == atype
    assert c.dtype == dtype
    assert c.sum() == 0


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("atype", ["numpy"])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "float32", "float64"])
def test_ones(a, b, atype, dtype):
    c = ones((a, b), atype=atype, dtype=dtype)

    assert c.shape == (a, b)
    assert find_package(c)[1] == atype
    assert c.dtype == dtype
    assert c.sum() == a * b


def test_matmul_overwrite():
    a = numpy.random.rand(5, 5)
    id_a_check = id(a)
    a = matmul(a, a, out=a)

    assert id_a_check == id(a)


def test_matmul_write():
    a = numpy.random.rand(5, 5)
    b = scipy.sparse.csc_matrix(a)
    id_a_check, id_b_check, id_b_data_check = id(a), id(b), id(b.data)
    a = matmul(a, a)
    b = matmul(b, b)

    assert id_a_check != id(a)
    assert id_b_check != id(b)
    assert id_b_data_check != id(b.data)


def test_matmul_equal():
    a = numpy.random.rand(5, 5)
    b = scipy.sparse.csc_matrix(a)
    prod_a = matmul(a.T, a)
    prod_b = matmul(b.T, b)

    assert (prod_a == prod_b).all() == True


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("atype", ["numpy", "scipy.sparse"])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "float32", "float64"])
def test_matmul(a, b, atype, dtype):
    c = zeros((a, b), atype, dtype)
    d = matmul(c.T, c)

    assert d.shape == (b, b)
    assert find_package(d)[1] == atype
    assert d.dtype == dtype
    assert d.sum() == 0


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("atype", ["numpy", "scipy.sparse"])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "float32", "float64"])
def test_multiply(a, b, atype, dtype):
    c = zeros((a, b), atype, dtype)
    d = multiply(c, c)

    assert d.shape == (a, b)
    assert find_package(d)[1] == atype
    assert d.dtype == dtype
    assert d.sum() == 0


def test_multiply_equal():
    a = numpy.random.rand(5, 5)
    b = scipy.sparse.csc_matrix(a)
    prod_a = matmul(a, a)
    prod_b = matmul(b, b)

    assert (prod_a == prod_b).all() == True


def test_multiply_overwrite():
    a = numpy.random.rand(5, 5)
    id_a_check = id(a)
    a = multiply(a, a, out=a)

    assert id_a_check == id(a)


def test_multiply_write():
    a = numpy.random.rand(5, 5)
    b = scipy.sparse.csc_matrix(a)
    id_a_check, id_b_check, id_b_data_check = id(a), id(b), id(b.data)
    a = multiply(a, a)
    b = multiply(b, b)

    assert id_a_check != id(a)
    assert id_b_check != id(b)
    assert id_b_data_check != id(b.data)


def test_astype():
    a = numpy.random.rand(5, 5)
    b = scipy.sparse.csc_matrix(a)

    a_astype = a.astype("float32")
    b_astype = b.astype("float32")

    assert (a_astype == b_astype).all() == True
    assert a_astype.shape == a.shape
    assert b_astype.shape == b.shape


def test_transpose():
    a = numpy.random.rand(10, 5)
    b = scipy.sparse.csc_matrix(a)

    a_transpose = transpose(a)
    b_transpose = transpose(b)

    assert (a_transpose == b_transpose).all() == True
    assert a_transpose.shape == a.shape[::-1]
    assert b_transpose.shape == b.shape[::-1]


def test_argsort():
    a = numpy.array([1, 5, 6, 8, 9])
    a_argsort = argsort(a)

    assert (a_argsort == numpy.linspace(0, a.size - 1, a.size)).all() == True
    assert a_argsort.shape == a.shape


def test_real():
    a = numpy.random.rand(10, 5)
    a_real = real(a)
    a_real2 = real(a + 1j * a)

    assert (a_real == a).all() == True
    assert a_real.shape == a.shape
    assert (a_real2 == a).all() == True
    assert a_real2.shape == a.shape


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_append(a, b):
    c = numpy.random.rand(a, b)
    c_append = append(c, c, axis=0)
    c_append1 = append(c, c, axis=1)

    assert c_append.shape[0] == 2 * c.shape[0]
    assert c_append.shape[1] == c.shape[1]
    assert c_append1.shape[0] == c.shape[0]
    assert c_append1.shape[1] == 2 * c.shape[1]


def test_array_any():
    assert array_any(numpy.zeros((5, 5))) == False
    assert array_any(numpy.random.rand(5, 5)) == True


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "float32", "float64"])
def test_rand(a, b, dtype):
    c = rand(a, b, atype="numpy", dtype=dtype)

    assert c.shape == (a, b)
    assert c.dtype == dtype


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "float32", "float64"])
def test_randn(a, b, dtype):
    c = randn(a, b, atype="numpy", dtype=dtype)

    assert c.shape == (a, b)
    assert c.dtype == dtype


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_absolute(a, b):
    c = numpy.random.rand(a, b)
    c_abs = abs(-c)

    assert c_abs.shape == c.shape
    assert isinstance(c_abs, type(c)) == True
    assert (c == c_abs).all() == True


@pytest.mark.parametrize("sparsity", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
def test_sparsify(sparsity):
    c = numpy.zeros((100, 100))
    c[:, :50] = numpy.random.rand(100, 50)
    sparsity_check = False if sparsity <= 0.5 else True
    d = scipy.sparse.csc_matrix(c)

    assert isinstance(sparsify(c, sparsity), scipy.sparse.csc_matrix) == sparsity_check
    assert isinstance(sparsify(d, sparsity), scipy.sparse.csc_matrix) == True


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("axis", [0, 1])
def test_sum(a, b, axis):
    c = numpy.random.rand(a, b)
    d = scipy.sparse.csc_matrix(c)

    assert sum(c, axis=axis).shape == (c.shape[1 - axis],)
    assert sum(d, axis=axis).shape == (d.shape[1 - axis],)


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("axis", [0, 1])
def test_sum_keepdims(a, b, axis):
    c = numpy.random.rand(a, b)

    assert sum(c, axis=axis, keepdims=True).ndim == 2


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_sum_total(a, b):
    c = numpy.ones((a, b))
    d = scipy.sparse.csc_matrix(c)

    assert sum(c, axis=None) == a * b
    assert sum(c) == a * b
    assert sum(d, axis=None) == a * b
    assert sum(d) == a * b


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_sign(a, b):
    c = numpy.ones((a, b))
    d = -c

    assert (c == sign(c)).all() == True
    assert (d == sign(d)).all() == True


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("axis", [0, 1])
def test_amax(a, b, axis):
    c = numpy.random.rand(a, b)
    d = scipy.sparse.csc_matrix(c)

    assert amax(c, axis=axis).shape == (c.shape[1 - axis],)
    assert (
        amax(d, axis=axis).shape == (1 - axis, d.shape[1 - axis], axis)[axis : axis + 2]
    )


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("axis", [0, 1, None])
@pytest.mark.parametrize("kind", ["quicksort", "mergesort", "heapsort", "stable"])
def test_sort(a, b, axis, kind):
    c = numpy.random.rand(a, b)

    assert (
        sort(c, axis=axis, kind=kind) == numpy.sort(c, axis=axis, kind=kind)
    ).all() == True


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("num", [10, 100, 1000])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_linspace(a, b, num, dtype):
    c = linspace(a, b, num, dtype=dtype)

    assert (c == numpy.linspace(a, b, num, dtype=dtype)).all() == True


@pytest.mark.parametrize("a", [2, 10, 100])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_roots(a, dtype):
    c = randn(a, dtype=dtype)

    assert (roots(c) == numpy.roots(c)).all() == True


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_imag(a, b):
    c = numpy.random.randn(a, b)
    d = c + 1j * c

    assert (c == imag(d)).all() == True
    assert imag(d).shape == c.shape


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("newshape", [(1, -1), (-1, 1), None])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_reshape(a, b, newshape, dtype):
    c = randn(a, b, dtype=dtype)

    assert (reshape(c, newshape) == c.reshape(newshape)).all() == True
    assert reshape(c, newshape).dtype == dtype


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
@pytest.mark.parametrize("axis", [0, 1, None])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_cumsum(a, b, axis, dtype):
    c = randn(a, b, dtype=dtype)

    assert (cumsum(c, axis=axis) == numpy.cumsum(c, axis=axis)).all() == True


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_maximum(a, b):
    c = numpy.ones((a, b))
    d = numpy.zeros((a, b))

    assert (c == maximum(c, d)).all() == True
    assert maximum(c, d).shape == c.shape


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_maximum(a, b):
    c = numpy.ones((a, b))
    d = numpy.zeros((a, b))

    assert (c == maximum(c, d)).all() == True
    assert maximum(c, d).shape == c.shape


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_minimum(a, b):
    c = numpy.ones((a, b))
    d = numpy.zeros((a, b))

    assert (d == minimum(c, d)).all() == True
    assert minimum(c, d).shape == c.shape


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_max_total(a, b):
    c = numpy.zeros((a, b))
    c[0, 0] = 1
    d = scipy.sparse.csc_matrix(c)

    assert amax(c, axis=None) == 1
    assert amax(c) == 1
    assert amax(d, axis=None) == 1
    assert amax(d) == 1


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test__numpy_to_sparse(a, b):
    c = numpy.random.rand(a, b)
    d = _numpy_to_sparse(c)

    assert find_package(d)[1] == "scipy.sparse"
    assert d.shape == c.shape
    assert (d.toarray() == c).all() == True


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test__sparse_to_numpy(a, b):
    c = numpy.random.rand(a, b)
    d = scipy.sparse.csc_matrix(c)
    d_out = _sparse_to_numpy(d)

    assert find_package(d_out)[1] == "numpy"
    assert d_out.shape == c.shape
    assert (d_out == c).all() == True


@pytest.mark.parametrize("a", [1, 10, 100])
@pytest.mark.parametrize("b", [1, 10, 100])
def test_divide(a, b):
    c = ones((a, b))
    d = divide(c, c)

    assert d.shape == (a, b)
    assert d.sum() == a * b


def test_divide_overwrite():
    a = numpy.random.rand(5, 5)
    id_a_check = id(a)
    print(divide)
    a = divide(a, a, out=a)

    assert id_a_check == id(a)


def test_divide_write():
    a = numpy.random.rand(5, 5)
    id_a_check = id(a)
    a = divide(a, a)

    assert id_a_check != id(a)
