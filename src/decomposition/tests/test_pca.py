from ...math import *
from ...math import linalg
from ..pca import pca


def test_pca_init():
    """Returns PCA with rank 1 (initialization)"""
    pca_object = pca()

    assert pca_object.rank == 1


def test_pca(n=500, rank_factor=0.10, card=0.05):
    rank = int(rank_factor * n)
    # Create low-rank with random bases
    x = (numpy.random.randn(n, rank)).astype(numpy.float32)
    y = (numpy.random.randn(n, rank)).astype(numpy.float32)
    l = y @ x.T

    # Create uniform noise pattern (m = l + s)
    s = numpy.random.uniform(low=-1, high=1, size=(n, n)).astype(numpy.float32)

    m = l + s

    # Run PCA
    pca_object = pca(verbose=False, rank=rank)
    pca_object.run(m)

    frob_l = linalg.norm(pca_object.b - l, "fro") / linalg.norm(l, "fro")

    errors = []

    # replace assertions by conditions
    if not frob_l <= 5:
        errors.append("Low-rank term not correct")

    # assert no error message has been registered, else print messages
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
