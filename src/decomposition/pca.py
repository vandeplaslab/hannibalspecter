import time as time
from typing import NoReturn, Union

from ..math import *
from ..math import linalg


class pca:
    """
    Perform principal component analysis, first preprocess:
    a = b + c, with b = tp^T

    Parameters
    ----------
    rank: int, default=1
        Rank of low-rank array b

    method : str, default='scipy_gesdd'
        See `svd_input` in 'math/linalg.py' for accepted methods

    verbose: bool, default=Truea
        Print option

    Attributes
    ----------
    run: array
        Run PCA algorithm on data set

    """

    def __init__(
        self,
        rank: int = 1,
        verbose: bool = True,
        sparsity: float = 0.1,
        **kwargs,
    ) -> NoReturn:
        self.rank = rank
        self._verbose = verbose
        self.sparsity = sparsity
        self.kwargs = kwargs

        pass

    @property
    def t(self) -> array:
        """
        Return t (q = tp^T)

        Returns
        -------
        t: numpy ndarray (2d)
            Score matrix

        """
        return multiply(self._b[0], self._b[1])

    @property
    def p(self) -> array:
        """
        Return p (q = tp^T)

        Returns
        -------
        p: numpy ndarray (2d)
            Loading matrix

        """
        return transpose(self._b[2])

    @property
    def b(self) -> array:
        """Reconstruct b (low-rank component) from svd components

        Returns
        -------
            array : reconstructed low-rank component

        """
        return matmul(self.t, transpose(self.p))

    @b.setter
    def b(self, a: tuple[array]) -> None:
        """Sets b from tuple

        Parameters
        ----------
        a : tuple[array]
            Tuple containing matrices of svd to set b

        """
        self._b[0] = a[0]
        self._b[1] = a[1]
        self._b[2] = a[2]

        return None

    @property
    def c(self) -> array:
        """Return c (a = b + c)

        Returns
        -------
        c : array
            Residual array

        """
        return sparsify(self.a - self.b, self.sparsity)

    def _initialize_b(self) -> None:
        """Initialize b"""
        self._b = [0, 0, 0]
        self._b[0] = zeros(
            (self._m, 1), atype=find_package(self.a)[1], dtype=self.datatype
        )
        self._b[1] = zeros((1, 1), atype=find_package(self.a)[1], dtype=self.datatype)
        self._b[2] = zeros(
            (1, self._n), atype=find_package(self.a)[1], dtype=self.datatype
        )

        return None

    def run(self, a: array) -> None:
        """Run PCA algorithm on data set

        Parameters
        ----------
        a : array
            Input array

        """
        self._tic = 0
        self.a = a
        self.datatype = a.dtype
        self._m, self._n = self.a.shape
        self._initialize_b()
        self._tic = time.time()  # Start timer
        self.kwargs.update({"sv": self.rank})
        self.kwargs.update(
            {"v0": self._b[0][0, :] if sum(self._b[0][:, 0]) > 1e-2 else None}
        )
        self.b = linalg.svd(self.a, self.kwargs)
        self._tic = time.time() - self._tic  # Finish timer
        if self._verbose:  # Print option
            self._print_out()

        return None

    def _print_out(self):
        """Output duration of PCAs"""
        text = ("PCA: done, time = {}s").format(
            int(self._tic),
        )
        print(text)

        return None
