import builtins
import math
import time as time
from typing import List, NoReturn, Union

from ..criteria import criteria
from unipy import *


class pcp:
    """
    Perform principal component pursuit (see [1]):
        a = b + c
    where
        a = input term
        b = low-rank term (u*s*vt)
        c = sparse term

    Parameters
    ----------
    criteria_list : Union[list, None], default=None
        List containing criteria, each item (a.k.a. criteria) can be a list or a simple string, e.g.
            [['$a$', '<', '0.05']] or [['$a$', '<', '0.05'], ['linalg.norm($a$,2)+linalg.norm($b$, 2)', '<', '0.05']].
        variables need to be put between dollar signs, i.e. $.$. Note that the first criterium (that is set in init) is required for `_update_mu()`.

    condition : {'any', 'all'}, default='all'
        Condition for the list of criteria, for 'any' check returns `True`
        if one or more critera are met. If 'all', check returns `True` if all critera are met.

    method : str, default='scipy_gesdd'
        See `svd_input` in 'math/linalg.py' for accepted methods

    maxiter : int, default=50
        Maximal number of iterations

    sv_factor : float [0-1], default=0.05
        Singular value factor to add in order to use truncated svd (see _update_svd)

    lambda_factor : float, default=1
        Setting factor to be multiplied with standard lambda: 1/sqrt(n) (see [1])

    lambda_opt : float, default=None
        Setting lambda automatically to preferred value

    verbose : bool, default=True
        Intermediate results printing

    Attributes
    ----------
    a :
        Input term

    b :
        Low-rank term

    c :
        Sparse term

    d :
        s_hat sparse term

    e :
        Residual term

    run : numpy ndarray (2d)
        Run principal component pursuit

    References
    ----------
    [1] Candès, Emmanuel J., et al. "Robust principal component analysis?."
    Journal of the ACM (JACM) 58.3 (2011): 1-37.

    [2] Lin, Zhouchen, Minming Chen, and Yi Ma. "The augmented lagrange
    multiplier method for exact recovery of corrupted low-rank matrices."
    arXiv preprint arXiv:1009.5055 (2010).

    """

    def __init__(
        self,
        criteria_list: Union[List, None] = None,
        condition: str = "all",
        maxiter: int = 50,
        sv_factor: float = 0.05,
        lambda_factor: float = 1,
        lambda_opt: float = None,
        verbose: bool = True,
        sparsity: float = 0.1,
        **kwargs,
    ) -> NoReturn:
        criteria_list = (
            criteria_list
            if isinstance(criteria_list, list) is True
            else [
                ["linalg.norm($a$-$b$-$d$, 'fro')/$a_fro$", "<", "1e-6"],
                ["$mu$*linalg.norm($c$-$d$, 'fro')/$a_fro$", "<", "1e-5"],
            ]
        )
        self.criteria = criteria(criteria_list, condition)
        self.maxiter = maxiter
        self.sv_factor = sv_factor
        self.lambda_factor = lambda_factor
        self.lambda_opt = lambda_opt
        self.verbose = verbose
        self.sparsity = sparsity
        self.kwargs = kwargs

        pass

    @property
    def b(self) -> array:
        """Reconstruct b (low-rank component) from svd components

        Returns
        -------
            array : reconstructed low-rank component

        """
        return matmul(multiply(self._b[0], self._b[1]), self._b[2])

    @property
    def e(self) -> array:
        """Reconstruct optimization residual (e=a-b-c)

        Returns
        -------
            array : residual

        """
        return self.a - self.b - self.c

    @b.setter
    def b(self, a: tuple[array]) -> None:
        """Sets b from tuple

        Parameters
        ----------
        a : tuple[array]
            Tuple containing arrays of svd to set b

        """
        self._b[0] = a[0]
        self._b[1] = a[1]
        self._b[2] = a[2]

        return None

    def _initialize_lambda(self) -> None:
        """Initialize lambda depending on its setting, lambda_opt is overwriting"""
        if self.lambda_opt is None:
            self.lambda_opt = (
                self.lambda_factor * 1 / math.sqrt(builtins.max(self._m, self._n))
            )
        else:
            self.lambda_factor = (
                1 / math.sqrt(builtins.max(self._m, self._n)) / self.lambda_opt
            )
        return None

    def _initialize_y(self) -> None:
        """Initialize y"""
        self._y = (
            1
            / builtins.max(
                self._a_norm_2,
                (1 / self.lambda_opt) * amax(self.a),
            )
        ) * self.a.astype(self.datatype)

        return None

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
    
    def _datatype(self, dtype: str) -> str:
        """Returns datatype and promotes if required"""
        dtype = dtype if dtype not in ['int8', 'int16', 'int32', 'int64'] else 'float32'
        
        return dtype
        
    
    def _initialize(self, a: array) -> None:
        """Initialize other help variables

        Parameters
        ----------
        a :  array
            Input array to be decomposed

        """
        self.a = a
        self.datatype = self._datatype(str(a.dtype))
        self._m, self._n = self.a.shape
        # Initialize variables
        self.atype = (
            "scipy.sparse"
            if find_package(self.a)[1] == "numpy"
            else "cupyx.scipy.sparse"
        )
        self.c = zeros((self._m, self._n), atype=self.atype, dtype=self.datatype)
        self.d = zeros((self._m, self._n), atype=self.atype, dtype=self.datatype)
        # Initialize Attributes
        self._a_norm_fro = linalg.norm(self.a, "fro")
        self._a_norm_2 = linalg.norm(self.a, 2)
        self._mu = 1.25 / self._a_norm_2
        self._rho = 1.6
        self._initialize_lambda()
        self._initialize_y()
        self._initialize_b()
        # Initialize Class Attributes
        self._k = 0  # Set iteration counter
        self._sv = 10  # set sv
        self.b_rank = [0]  # Rank change over iterations
        self.c_card = [0]  # Cardinality change over iterations
        self._tic = 0  # Initialize counter
        self._x = {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "y": self._y,
            "a_2": self._a_norm_2,
            "a_fro": self._a_norm_fro,
            "mu": self._mu,
        }  # Dictionary with references for criteria
        return None

    def run(self, a: array) -> None:
        """Run algorithm

        Parameters
        ----------
        a : array
            Input array to be decomposed

        """
        self._initialize(a)
        self._tic = time.time()
        self.check = self.criteria.check(self._x)
        while (self.check is False) and (self._k < self.maxiter):
            self._update_d()
            self._update_b()
            self._update_sv()
            self._update_y()
            self._x = {
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "d": self.d,
                "y": self._y,
                "a_2": self._a_norm_2,
                "a_fro": self._a_norm_fro,
                "mu": self._mu,
            }  # Dictionary with references for criteria
            self.check = self.criteria.check(self._x)
            self._update_mu()
            self._update_c()
            self._update_parameters()
            self._update_k()
            if self.verbose:
                self._print_out()

        self._tic = time.time() - self._tic  # Finish timer

        return None

    def _update_c(self) -> None:
        """Update c by d (estimate)"""       
        if copyto(self.c, self.d) is False:
            self.c =  astype(self.d, dtype=self.d.dtype, copy=True)
        
        return None

    def _update_b(self) -> None:
        """Update b by performing truncated svd"""
        if "method" in self.kwargs:
            sv = (
                self._sv + 10 if linalg.requires_sv(self.kwargs["method"]) is True else 0
            )
        else:
            sv = 0

        self.kwargs.update({"sv": sv})
        self.kwargs.update(
            {"v0": self._b[2] if sum(self._b[1]) > 1e-2 else None}
        )
        
        self.b = linalg.svd(self.a - self.d + (1 / self._mu) * self._y, self.kwargs)
        self._svp = int(sum(self._b[1] > 1 / self._mu))
        if self._svp != 0:
            self._threshold(self._svp, 1 / self._mu)
        else:  # if svp is 0, just reset the svd result
            self._initialize_b()

        return None

    def _update_d(self) -> None:
        """Update d (estimate)"""
        if copyto(self.d, sparsify(
            soft_threshold(
                self.a - self.b + (1 / self._mu) * self._y,
                (1 / self._mu) * self.lambda_opt,
            ),
            self.sparsity,
        )) is False:
            self.d = sparsify(
                soft_threshold(
                    self.a - self.b + (1 / self._mu) * self._y,
                    (1 / self._mu) * self.lambda_opt,
                ),
                self.sparsity,
            )

        return None

    def _update_y(self) -> None:
        """Update y"""
        self._y += self._mu * (self.a - self.b - self.d)

        return None

    def _update_mu(self) -> None:
        """Update mu parameter"""
        a = "linalg.norm($a$-$b$-$d$, 'fro')/$a_fro$"
        criteria_listed = [
            item for sublist in self.criteria.criteria for item in sublist
        ]
        try:
            index = criteria_listed.index(a)
            if self._mu * self.criteria.state[index][0] < self.criteria.state[index][2]:
                self._mu = self._rho * self._mu

        except:
            self._mu = self._mu

        return None

    def _update_sv(self) -> None:
        """Update (predict) the number of singular values to be calculated by svd"""
        n = min(self._m, self._n)
        if self._svp < self._sv:
            self._sv = int(min(self._svp + 1, n))
        else:
            self._sv = int(min(self._svp + round(self.sv_factor * n), n))
            
        return None

    def _threshold(self, num: int = 1, value: float = 0.0) -> None:
        """Threshold b to certain value

        Parameters
        ----------
        num: int
            Truncate to this number of values

        value : float
            Threshold to this value

        """
        self._b[0] = self._b[0][:, :num]
        self._b[1] = self._b[1][:num] - value
        self._b[2] = self._b[2][:num, :]

        return None

    def _update_parameters(self) -> None:
        """Update the low-rank matrix rank and cardinality of c (each iteration) and update dictionary for criteria"""
        self.b_rank.append(self._svp)
        self.c_card.append(count_nonzero(self.c))

        return None

    def _update_k(self) -> None:
        """Update k (iteration number)"""
        self._k += 1

        return None

    def _print_out(self) -> None:
        """Output iteration steps during optimization"""
        text = (
            "#{} - Rank = {}% ({}) - Card = {}% ({})" " - sv_[k+1] = {} - time = {}s"
        ).format(
            self._k,
            round(100 * self.b_rank[-1] / min(self._m, self._n), 1),
            self.b_rank[-1],
            round(100 * self.c_card[-1] / (self._m * self._n), 1),
            self.c_card[-1],
            self.kwargs['sv'],
            int(time.time() - self._tic),
        ) + "".join(
            [
                " - crit" + str(i) + " = " + str(self.criteria.state[i][0])
                for i in range(len(self.criteria.state))
            ]
        )
        print(text)

        return None
