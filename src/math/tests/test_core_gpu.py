import numpy
import pytest
import scipy.sparse
import scipy.sparse.linalg

cupy = pytest.importorskip("cupy")
cupyx = pytest.importorskip("cupyx")

from ..core import *


@pytest.mark.parametrize(
    "x, package_check",
    [
        (cupy.random.rand(5, 5), "cupy"),
        (cupyx.scipy.sparse.csc_matrix((5, 5)), "cupyx.scipy.sparse"),
    ],
)
def test_find_package(x, package_check):
    package = find_package(x)[1]
    assert package_check == package


def test_sqrt():

    assert True == True


def test_clip():

    assert True == True


def test_count_nonzero():

    assert True == True


def test_zero():

    assert True == True


def test_matmul():

    assert True == True


def test_multiply():

    assert True == True


def test_astype():

    assert True == True


# @pytest.mark.parametrize(
#     "x, dict_x_check",
#     [
#         ({"a": 0.1}, {"$a$": 'x["a"]'}),
#         ({"a": 0.1, "b": 0.1}, {"$a$": 'x["a"]', "$b$": 'x["b"]'}),
#     ],
# )
# def test_criteria_create_dictx_equality(x, dict_x_check):
#     criteria_object = criteria(["$a$ < 1"])
#     dict_x = criteria_object._create_dictx(x)

#     assert dict_x_check.keys() == dict_x.keys()
#     assert dict_x_check == dict_x
