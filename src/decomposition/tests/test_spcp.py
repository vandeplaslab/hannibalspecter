import numpy

from ...math import *
from ...math import linalg
from ..spcp import spcp


def test_spcp_init():
    spcp_object = spcp()

    assert spcp_object.lambda_factor == 1


def test_spcp(n=500, rank_factor=0.10, card=0.05, rho=0.05):
    rank = int(rank_factor * n)
    # Create low-rank with random bases
    x = (numpy.random.randn(n, rank)).astype(numpy.float32)
    y = (numpy.random.randn(n, rank)).astype(numpy.float32)
    l = y @ x.T

    # Create uniform noise pattern (m = l + s)
    s = numpy.random.binomial(1, card, (n, n)).astype(
        numpy.float32
    ) * numpy.random.uniform(low=-500, high=500, size=(n, n)).astype(numpy.float32)

    # Create small Gaussian noise
    e = (rho**2) * (numpy.random.rand(n, n)).astype(numpy.float32)

    m = l + s + e

    # delta
    delta = numpy.sqrt((n + numpy.sqrt(8 * n))) * rho

    # Run PCP
    spcp_object = spcp(dirac=delta, verbose=False)
    spcp_object.run(m)

    frob_l = linalg.norm(spcp_object.b - l, "fro") / numpy.linalg.norm(l, "fro")
    frob_s = linalg.norm(spcp_object.c - s, "fro") / numpy.linalg.norm(s, "fro")

    errors = []

    # replace assertions by conditions
    if not frob_l <= 0.005:
        errors.append("Low-rank term not correct")
    if not frob_s <= 0.005:
        errors.append("Sparse term not correct")

    # assert no error message has been registered, else print messages
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
