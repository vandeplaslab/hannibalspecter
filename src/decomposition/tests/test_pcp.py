import numpy

from ...math import *
from ...math import linalg
from ..pcp import pcp


def test_pcp_init():
    pcp_object = pcp()

    assert pcp_object.lambda_factor == 1


def test_pcp(n=500, rank_factor=0.10, card=0.05):
    rank = int(rank_factor * n)
    # Create low-rank with random bases
    x = (numpy.random.randn(n, rank)).astype(numpy.float32)
    y = (numpy.random.randn(n, rank)).astype(numpy.float32)
    l = y @ x.T

    # Create uniform noise pattern (m = l + s)
    s = numpy.random.binomial(1, card, (n, n)).astype(
        numpy.float32
    ) * numpy.random.uniform(low=-500, high=500, size=(n, n)).astype(numpy.float32)

    m = l + s

    # Run PCP
    pcp_object = pcp(verbose=False)
    pcp_object.run(m)

    frob_l = linalg.norm(pcp_object.b - l, "fro") / linalg.norm(l, "fro")
    frob_s = linalg.norm(pcp_object.c - s, "fro") / linalg.norm(s, "fro")

    errors = []

    # replace assertions by conditions
    if not frob_l <= 0.005:
        errors.append("Low-rank term not correct")
    if not frob_s <= 0.005:
        errors.append("Sparse term not correct")

    # assert no error message has been registered, else print messages
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
