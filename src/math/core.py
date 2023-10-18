import functools
import numbers
import warnings
from typing import Union

import numpy
import scipy.sparse
import scipy.sparse.linalg

array = Union[numpy.ndarray, scipy.sparse.spmatrix]
min_array = numpy.ndarray
sparse_array = scipy.sparse.spmatrix
gpu_array = None
gpu_min_array = None
gpu_sparse_array = None

try:
    import cupy
    import cupyx
    import cupyx.scipy.sparse.linalg

    array = Union[
        numpy.ndarray,
        cupy.ndarray,
        scipy.sparse.spmatrix,
        cupyx.scipy.sparse.spmatrix,
    ]
    min_array = Union[
        numpy.ndarray,
        cupy.ndarray,
    ]
    sparse_array = (scipy.sparse.spmatrix, cupyx.scipy.sparse.spmatrix)
    gpu_array = Union[
        cupy.ndarray,
        cupyx.scipy.sparse.spmatrix,
    ]
    gpu_min_array = cupy.ndarray
    gpu_sparse_array = cupyx.scipy.sparse.spmatrix

except ImportError as e:
    print(
        "Note: cupy and/or cupyx cannot be found, they are not necessary to use Hannibal Specter."
    )
    pass


type_class = {
    numpy.ndarray: "numpy",
    numpy.matrix: "numpy",
    scipy.sparse.spmatrix: "scipy.sparse",
    cupy._core.core.ndarray: "cupy",
    cupyx.scipy.sparse.spmatrix: "cupyx.scipy.sparse",
}

short_name_class = {
    "numpy": "numpy",
    "scipy.sparse": "sparse",
    "cupy": "cupy",
    "cupyx.scipy.sparse": "cupy_sparse",
}


def absolute(a: array, out: Union[array, None] = None) -> array:
    """Return the element-wise absolute value of an array.

    Parameters
    ----------
    a : array
        The values whose square-roots are required

    out : Union[array, None], default=None
        Location into which the array is stored

    Returns
    -------
        array : The values of absolute value
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.absolute(a, out=out)
    elif hs_math == "scipy.sparse":
        a.data = numpy.absolute(a.data, out=out)
        return a
    elif hs_math == "cupy":
        return cupy.absolute(a, out=out)
    elif hs_math == "cupyx.scipy.sparse":
        a.data = cupy.absolute(a, out=out)
        return a


def sqrt(a: array, out: Union[array, None] = None) -> array:
    """Return the non-negative square-root of an array, element-wise.

    Parameters
    ----------
    a : array
        The values whose square-roots are required

    out : Union[array, None], default=None
        Location into which the array is stored

    Returns
    -------
        array : The values of square-roots
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.sqrt(a, out=out)
    elif hs_math == "scipy.sparse":
        a.data = numpy.sqrt(a.data, out=out)
        return a
    elif hs_math == "cupy":
        return cupy.sqrt(a, out=out)
    elif hs_math == "cupyx.scipy.sparse":
        a.data = cupy.sqrt(a, out=out)
        return a


def clip(
    a: array,
    b: Union[int, float, None],
    c: Union[int, float, None],
    out: Union[array, None] = None,
) -> array:
    """Clip (limit) the values in an array

    Parameters
    ----------
    a : array
        Array to be clipped

    b: Union[int, float]
        Minimum value

    c: Union[int, float]
        Maximum value

    out : Union[array, None] default=None
        Location into which the array is stored

    Returns
    -------
        array : Clipped array

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.clip(a, b, c, out=out)
    elif hs_math == "scipy.sparse":
        a.data = numpy.clip(a.data, b, c, out=out)
        return a
    elif hs_math == "cupy":
        return cupy.clip(a, b, c, out=out)
    elif hs_math == "cupyx.scipy.sparse":
        a.data = cupy.clip(a.data, b, c, out=out)
        return a


def count_nonzero(a: array) -> int:
    """Count nonzero entries in array

    Parameters
    ----------
    a : array
        Input array

    Returns
    -------
        int : Number of nonzero entries

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.count_nonzero(a)
    elif hs_math == "scipy.sparse":
        return a.count_nonzero()
    elif hs_math == "cupy":
        return int(cupy.count_nonzero(a))
    elif hs_math == "cupyx.scipy.sparse":
        return int(a.count_nonzero())

def eye(shape: tuple, atype: str = "numpy", dtype: str = "float32") -> array:
    """Create array of ones on diagonal (Identity)

    Parameters
    ----------
    shape : tuple
        shape

    atype : str
        Array type with options: {'numpy', 'scipy.sparse', 'cupy', 'cupyx.scipy.sparse'}

    dtype : str
        Data type, e.g. 'float16', 'float32', 'int16', ...

    Returns
    -------
        array : Array containing zeros

    """
    if atype == "numpy":
        return numpy.eye(shape, dtype=dtype)
    elif atype == "scipy.sparse":
        return scipy.sparse.eye(shape, dtype=dtype)
    elif atype == "cupy":
        return cupy.eye(shape, dtype=dtype)
    elif atype == "cupyx.scipy.sparse":
        return cupyx.scipy.sparse.eye(shape, dtype=dtype)
    
def zeros(shape: tuple, atype: str = "numpy", dtype: str = "float32") -> array:
    """Create array of zeros

    Parameters
    ----------
    shape : tuple
        shape

    atype : str
        Array type with options: {'numpy', 'scipy.sparse', 'cupy', 'cupyx.scipy.sparse'}

    dtype : str
        Data type, e.g. 'float16', 'float32', 'int16', ...

    Returns
    -------
        array : Array containing zeros

    """
    if atype == "numpy":
        return numpy.zeros(shape, dtype=dtype)
    elif atype == "scipy.sparse":
        return scipy.sparse.csc_array(shape, dtype=dtype)
    elif atype == "cupy":
        return cupy.zeros(shape, dtype=dtype)
    elif atype == "cupyx.scipy.sparse":
        return cupyx.scipy.sparse.csc_matrix(shape, dtype=dtype)


def ones(shape: tuple, atype: str = "numpy", dtype: str = "float32") -> array:
    """Create array of ones

    Parameters
    ----------
    shape : tuple
        shape

    atype : str
        Array type with options: {'numpy', 'cupy'}

    dtype : str
        Data type, e.g. 'float16', 'float32', 'int16', ...

    Returns
    -------
        array : Array containing ones

    """
    if atype == "numpy":
        return numpy.ones(shape, dtype=dtype)
    elif atype == "cupy":
        return cupy.ones(shape, dtype=dtype)
    else:
        raise Exception("ONES not found for " + atype)


def linspace(
    a: int, b: int, num: int, atype: str = "numpy", dtype: str = "float32"
) -> array:
    """Return evenly spaced numbers over a specified interval.

    Parameters
    ----------
    a : int
        The starting value of the sequence.

    b : int
        The end value of the sequence

    num : int
        Number of samples to generate

    atype : str
        Array type with options: {'numpy', 'cupy'}

    dtype : str
        Data type, e.g. 'float16', 'float32', 'int16', ...

    Returns
    -------
        array : Array containing linspace

    """
    if atype == "numpy":
        return numpy.linspace(a, b, num, dtype=dtype)
    elif atype == "cupy":
        return cupy.linspace(a, b, num, dtype=dtype)
    else:
        raise Exception("LINSPACE not found for " + atype)


def cumsum(
    a: array, axis: Union[int, None] = None, out: Union[array, None] = None
) -> array:
    """Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : array
        Input array

    axis : Union[int, None], default=None
        Axis along which to sum

    out : array
        Location into which the array is stored

    Returns
    -------
        array : Cumulatived summed array

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.cumsum(a, axis=axis, out=out)
    elif hs_math == "cupy":
        return cupy.cumsum(a, axis=axis, out=out)
    else:
        raise Exception("CUMSUM not found for " + hs_math)


def reshape(a: array, newshape: Union[int, tuple]) -> array:
    """Gives a new shape to an array without changing its data.

    Parameters
    ----------
    a : array
        Input array

    newshape : Union[int, tuple]
        The new shape should be compatible with the original shape.

    Returns
    -------
        array : Reshaped array

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.reshape(a, newshape)
    elif hs_math == "cupy":
        return cupy.reshape(a, newshape)
    else:
        raise Exception("RESHAPE not found for " + hs_math)


def roots(a: array) -> array:
    """Return the roots of a polynomial with coefficients given in p.

    Parameters
    ----------
    a : array
        Input array

    Returns
    -------
        array : Roots

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.roots(a)
    elif hs_math == "cupy":
        return cupy.roots(a)
    else:
        raise Exception("ROOTS not found for " + hs_math)


def imag(a: array) -> array:
    """Return the imaginary part of the complex argument.

    Parameters
    ----------
    a : array
        Input array

    Returns
    -------
        array : Imaginary part of elements of input array

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.imag(a)
    elif hs_math == "cupy":
        return cupy.imag(a)
    else:
        raise Exception("IMAG not found for " + hs_math)


def matmul(a: array, b: array, out: Union[array, None] = None) -> array:
    """Matrix product of two matrices

    Parameters
    ----------
    a : array
        Input array

    b : array
        Input array

    out : array
        Location into which the array is stored

    Returns
    -------
        array : Matrix product of the two matrices

    """
    a, b, hs_math = find_package(a, b)

    if hs_math == "numpy":
        return numpy.matmul(a, b, out=out)
    elif hs_math == "scipy.sparse":
        return a.dot(b)
    elif hs_math == "cupy":
        return cupy.matmul(a, b, out=out)
    elif hs_math == "cupyx.scipy.sparse":
        return a.dot(b)


def multiply(a: array, b: array, out: Union[array, None] = None) -> array:
    """Element-wise multiply two matrices

    Parameters
    ----------
    a : array
        Input array

    b : array
        Input array

    Returns
    -------
        array : Element-wise multiplied array

    """
    a, b, hs_math = find_package(a, b)

    if hs_math == "numpy":
        return numpy.multiply(a, b, out=out)
    elif hs_math == "scipy.sparse":
        return a.multiply(b)
    elif hs_math == "cupy":
        return cupy.multiply(a, b, out=out)
    elif hs_math == "cupyx.scipy.sparse":
        return a.multiply(b)


def divide(a: array, b: array, out: Union[array, None] = None) -> array:
    """Element-wise divide two matrices

    Parameters
    ----------
    a : array
        Input array (nominator)

    b : array
        Input array (denominator)

    Returns
    -------
        array : Element-wise divided array

    """
    a, b, hs_math = find_package(a, b)

    if hs_math == "numpy":
        return numpy.divide(a, b, out=out)
    elif hs_math == "cupy":
        return cupy.divide(a, b, out=out)
    else:
        raise Exception("DIVIDE not found for " + hs_math)


def astype(
    a: array,
    dtype: str,
    copy: bool = False,
) -> array:
    """Redefine datatype of array and/or create a copy in memory.
    Note: cupyx.scipy.sparse always returns a copy

    Parameters
    ----------
    a : array
        Input Array

    dtype : str
        Datatype e.g. with options: {'float16', 'float32', 'int16', 'int32'}

    copy : bool, default=False
        Boolean defining whether a copy should be made in memory

    Returns
    -------
        array : (Copied) array with desired datatype

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return a.astype(dtype, copy=copy)
    elif hs_math == "scipy.sparse":
        return a.astype(dtype, copy=copy)
    elif hs_math == "cupy":
        return a.astype(dtype, copy=copy)
    elif hs_math == "cupyx.scipy.sparse":
        return a.astype(dtype)


def transpose(a: array) -> array:
    """Transposes input array

    Parameters
    ----------
    a : array
       Input array

    Returns
    -------
        array : Transposed array
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.transpose(a)
    elif hs_math == "scipy.sparse":
        return a.T
    elif hs_math == "cupy":
        return cupy.transpose(a)
    elif hs_math == "cupyx.scipy.sparse":
        return a.T


def argsort(a: min_array) -> min_array:
    """Sorts input array

    Parameters
    ----------
    a : min_array
       Input array

    Returns
    -------
        min_array : Sorted array
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.argsort(a)
    elif hs_math == "cupy":
        return cupy.argsort(a)
    else:
        raise Exception("ARGSORT not found for " + hs_math)


def real(a: min_array) -> min_array:
    """Retuns the real value of input array

    Parameters
    ----------
    a : min_array
       Input array

    Returns
    -------
        min_array : Real-valued array
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.real(a)
    elif hs_math == "cupy":
        return cupy.real(a)
    else:
        raise Exception("REAL not found for " + hs_math)


def append(a: min_array, b: min_array, axis: Union[int, None] = None) -> min_array:
    """Retuns the minimum value(s) of input array on axis

    Parameters
    ----------
    a : min_array
       Input array

    b : min_array
       Input array

    axis: Union[None, int], default=None
        Axis along which append has to be performed

    Returns
    -------
        min_array : Appended array
    """
    a, b, hs_math = find_package(a, b)

    if hs_math == "numpy":
        return numpy.append(a, b, axis=axis)
    elif hs_math == "cupy":
        return cupy.append(a, b, axis=axis)
    else:
        raise Exception("APPEND not found for " + hs_math)


def array_any(a: min_array) -> bool:
    """Checks if some values of input matrix are set (i.e. non-zero)

    Parameters
    ----------
    a : min_array
       Input array

    Returns
    -------
        bool : True if some values of input_array are different from 0, false if all values are 0.
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.any(a)
    elif hs_math == "cupy":
        return cupy.any(a)
    else:
        raise Exception("ARRAY_ANY not found for " + hs_math)


def rand(
    *args: int,
    atype: str = "numpy",
    dtype: str = "float32",
    random_state: Union[int, None] = None,
) -> min_array:
    """Returns an array of uniform random values over the interval [0, 1).

    Parameters
    ----------
    *args : int
        Integer defining array size

    atype : str
        Array type with options: {'numpy', 'cupy'}

    dtype : str
        Data type, i.e. 'float16', 'float32' or 'float64'

    random_state: Union[int, None], default=None
        Not yet implemented. Uses random state/seed to initialize random matrices

    Returns
    -------
       min_array : Array containing random samples

    """
    if atype == "numpy":
        return astype(numpy.random.rand(*args), dtype=dtype)
    elif atype == "cupy":
        return astype(cupy.random.rand(*args), dtype=dtype)
    else:
        raise Exception("RAND not found for " + atype)


def randn(
    *args: int,
    atype: str = "numpy",
    dtype: str = "float32",
    random_state: Union[int, None] = None,
) -> min_array:
    """Return a sample (or samples) from the “standard normal” distribution.

    Parameters
    ----------
    *args : int
        Integer defining array size

    atype : str
        Array type with options: {'numpy', 'cupy'}

    dtype : str
        Data type, i.e. 'float16', 'float32' or 'float64'

    random_state: Union[int, None], default=None
        Not yet implemented. Uses random state/seed to initialize random matrices

    Returns
    -------
        min_array : Array containing random samples

    """
    if atype == "numpy":
        return astype(numpy.random.randn(*args), dtype=dtype)
    elif atype == "cupy":
        return astype(cupy.random.randn(*args), dtype=dtype)
    else:
        raise Exception("RANDN not found for " + atype)


def amax(
    a: array, axis: Union[int, None, tuple] = None, out: Union[array, None] = None
) -> Union[array, float]:
    """Finds the largest element

    Parameters
    ----------
    a : array
        Input array

    axis : Union[int, None, tuple], default=None
        Axis or axes along which a sum is performed.

    out : Union[array, None], default=None
        Alternative output array in which to place the result.

    Returns
    -------
        Union[array, float] : Maximum value or values

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.amax(a, axis, out=out)
    elif hs_math == "scipy.sparse":
        return a.max(axis, out=out)
    elif hs_math == "cupy":
        return cupy.amax(a, axis, out=out)
    elif hs_math == "cupyx.scipy.sparse":
        return a.max(axis, out=out)


def sum(
    a: array,
    axis: Union[int, None, tuple] = None,
    out: Union[array, None] = None,
    keepdims: bool = False,
) -> Union[array, float]:
    """Sums along certain axis or axes of input array

    Parameters
    ----------
    a : array
        Input array

    axis : Union[int, None, tuple], default=None
        Axis or axes along which a sum is performed.

    out : Union[array, None], default=None
        Alternative output array in which to place the result.

    keepdims : bool, default=False
        Keep dimensions when norm is applied

    Returns
    -------
        Union[array, float] : Array or float as summed values along axis/axes

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.sum(a, axis, out=out, keepdims=keepdims)
    elif hs_math == "scipy.sparse":
        if isinstance(axis, tuple):
            raise Exception("SUM of sparse array does not accept tuples for axis")
        _q = a.sum(axis, out=out)
        if isinstance(_q, numpy.matrix):
            return _q.A1
        else:
            return _q
    elif hs_math == "cupy":
        return cupy.sum(a, axis, out=out, keepdims=keepdims)
    elif hs_math == "cupyx.scipy.sparse":
        return a.sum(axis, out=out)


def find_package(
    *args,
) -> tuple[numbers.Real, array, str]:
    """Searches for the corresponding linear algebra toolbox provided in type_class at the start of this file. Also gives the ability to change array types.

    Parameters
    ----------
    a : array
        Variable used for finding corresponding linear algebra toolbox

    Returns
    -------
        tuple(array, str) : Array containing input array (possible modification) and string containing the right pointer to the toolbox in string format

    """
    result = []
    atype = []
    i = 0
    for a in args:
        check = [isinstance(a, b) for b in type_class.keys()]
        if any(check) is True:
            if isinstance(a, numpy.matrix):
                result.append(a.A)
            else:
                result.append(a)
            atype.append(
                type_class[
                    list(type_class.keys())[[i for i, x in enumerate(check) if x][0]]
                ]
            )
        elif isinstance(a, numbers.Real):
            result.append(a)
            atype.append("real")
        else:
            raise Exception("Data type not supported.", type(a))

    result.append(_atype_list_to_string(atype))
    return tuple(result)


def _atype_list_to_string(atype: list[str]) -> str:
    """Converts list of strings containing array types to a single array type

    Parameters
    ----------
    atype : list[str]
        List of strings containing array types

    Returns
    -------
       str : Appropriate array type for operations

    """
    if atype.count("real") == len(atype):
        atype = "numpy"
    elif len(set(atype)) == 2 and "real" in list(set(atype)):
        atype_set = list(set(atype))
        atype = atype_set[0] if atype_set[0] != "real" else atype_set[1]
    elif all(elem == atype[0] for elem in atype):
        atype = atype[0]
    elif (
        atype.count("numpy") > 0
        and atype.count("cupy") == 0
        and atype.count("cupyx.scipy.sparse") == 0
    ):
        atype = "numpy"
    else:
        raise Exception(
            "Cannot define a particular package for this operation. This can be due to multiple incompatible inputs."
        )
    return atype


def sign(a: array, out: Union[array, None] = None) -> array:
    """Returns an element-wise indication of the sign of a number.

    Parameters
    ----------
    a : array
        Input array

    Returns
    -------
        array: sign array of input a
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.sign(a, out=out)
    elif hs_math == "cupy":
        return cupy.sign(a, out=out)
    else:
        raise Exception("SIGN not yet implemented for " + hs_math)


def maximum(
    a: Union[float, array],
    b: Union[float, array],
    out: Union[array, None] = None,
) -> Union[float, array]:
    """Element-wise maximum of array elements.

    Parameters
    ----------
    a : Union[float, array]
        Input array

    b : Union[float, array]
        Input array

    out : Union[array, None], default=None
        Alternative output array in which to place the result.

    Returns
    -------
        Union[float, array]: Maximum of a and b

    """
    a, b, hs_math = find_package(a, b)

    if hs_math == "numpy":
        return numpy.maximum(a, b, out=out)
    elif hs_math == "cupy":
        return cupy.maximum(a, b, out=out)
    else:
        raise Exception("MAXIMUM not yet implemented for " + hs_math)


def minimum(
    a: Union[float, array], b: Union[float, array], out: Union[array, None] = None
) -> Union[float, array]:
    """Element-wise minimum of array elements.

    Parameters
    ----------
    a : Union[float, array]
        Input array

    b : Union[float, array]
        Input array

    out : Union[array, None], default=None
        Alternative output array in which to place the result.

    Returns
    -------
        Union[float, array]: Minimum of a and b

    """
    a, b, hs_math = find_package(a, b)

    if hs_math == "numpy":
        return numpy.minimum(a, b, out=out)
    elif hs_math == "cupy":
        return cupy.minimum(a, b, out=out)
    else:
        raise Exception("MINIMUM not yet implemented for " + hs_math)


def sort(a: array, axis: Union[int, None] = None, kind: str = "quicksort") -> array:
    """Return a sorted copy of an array

    Parameters
    ----------
    a : array
        Array to be sorted

    axis : Union[int, None], default=None
        Axis along which to sort

    kind : {‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}
        Sorting algorithm

    Returns
    -------
        array: Sorted array

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.sort(a, axis=axis, kind=kind)
    elif hs_math == "cupy":
        return cupy.sort(a, axis=axis)
    else:
        raise Exception("SIGN not yet implemented for " + hs_math)


def sparsify(a: array, sparsity: float) -> array:
    """Returns sparse version of input array if number of non-zero entries is lower than sparsity

    Parameters
    ----------
    a : array
        Input array

    sparsity : float [0-1]
        Sparsity (between 0 and 1, where 1 corresponds to all entries to be non-zero) of the input array for which it needs to become sparse

    Returns
    -------
        array : Output array

    """
    if isinstance(a, sparse_array):
        return a
    elif count_nonzero(a) / (a.shape[0] * a.shape[1]) >= sparsity:
        return a
    else:
        atype = find_package(a)[1]
        if atype == "numpy":
            return _numpy_to_sparse(a)
        elif atype == "cupy":
            return _cupy_to_cupy_sparse(a)


def _numpy_to_sparse(a: numpy.ndarray) -> scipy.sparse.spmatrix:
    """Convert numpy array to sparse array

    Parameters
    ----------
    a : numpy.ndarray
        Dense input matrix

    Returns
    -------
        scipy.sparse.spmatrix: converted array

    """
    return scipy.sparse.csc_array(a)


def _sparse_to_numpy(a: scipy.sparse.spmatrix) -> numpy.ndarray:
    """Convert sparse array to dense numpy array

    Parameters
    ----------
    a : scipy.sparse.spmatrix
        Array to be converted

    Returns
    -------
        numpy.ndarray: converted array

    """
    return a.toarray()


def _numpy_to_cupy(a: numpy.ndarray) -> gpu_min_array:
    """Convert numpy array to cupy array

    Parameters
    ----------
    a : numpy.ndarray
        Array to be converted

    Returns
    -------
        gpu_min_array: converted array

    """
    return cupy.asarray(a)


def _cupy_to_numpy(a: gpu_min_array) -> numpy.ndarray:
    """Convert cupy array to numpy array

    Parameters
    ----------
    a : cupy.ndarray
        Array to be converted

    Returns
    -------
       numpy.ndarray : converted array

    """
    return cupy.asnumpy(a)


def _sparse_to_cupy_sparse(a: scipy.sparse.spmatrix) -> gpu_sparse_array:
    """Convert sparse array to cupy sparse array

    Parameters
    ----------
    a : scipy.sparse.spmatrix
        Array to be converted

    Returns
    -------
       gpu_sparse_array : converted array

    """
    return cupyx.scipy.sparse.c(a)


def _cupy_sparse_to_sparse(a: gpu_sparse_array) -> scipy.sparse.spmatrix:
    """Convert cupy sparse array to sparse array

    Parameters
    ----------
    a : gpu_sparse_array
        Array to be converted

    Returns
    -------
       scipy.sparse.spmatrix : converted array

    """
    return scipy.sparse.csc_array(
        (cupy.asnumpy(a.data), cupy.asnumpy(a.indices), cupy.asnumpy(a.indptr)),
        shape=a.shape,
    )


def _cupy_sparse_to_cupy(a: gpu_sparse_array) -> gpu_min_array:
    """Convert cupy sparse array to dupy dense array

    Parameters
    ----------
    a : gpu_sparse_array
        Array to be converted

    Returns
    -------
       gpu_min_array : converted array

    """
    return a.toarray()


def _cupy_to_cupy_sparse(a: gpu_min_array) -> gpu_sparse_array:
    """Convert cupy dense array to cupy sparse array

    Parameters
    ----------
    a : gpu_min_array
        Array to be converted

    Returns
    -------
       gpu_sparse_array : converted array

    """
    return cupyx.scipy.sparse.csc_matrix(a)


def _sparse_to_cupy(a: scipy.sparse.spmatrix) -> gpu_min_array:
    """Convert sparse array to cupy dnese array

    Parameters
    ----------
    a : scipy.sparse.spmatrix
        Array to be converted

    Returns
    -------
       gpu_min_array : converted array

    """
    return _cupy_sparse_to_cupy(_sparse_to_cupy_sparse(a))


def _cupy_to_sparse(a: gpu_min_array) -> scipy.sparse.spmatrix:
    """Convert cupy array to sparse array

    Parameters
    ----------
    a : gpu_min_array
        Array to be converted

    Returns
    -------
        scipy.sparse.spmatrix : converted array

    """
    return _cupy_sparse_to_sparse(_cupy_to_cupy_sparse(a))


def _cupy_sparse_to_numpy(a: gpu_sparse_array) -> numpy.ndarray:
    """Convert cupy sparse array to numpy array

    Parameters
    ----------
    a : gpu_sparse_array
        Array to be converted

    Returns
    -------
        numpy.ndarray : converted array

    """
    return _sparse_to_numpy(_cupy_sparse_to_sparse(a))


def _numpy_to_cupy_sparse(a: numpy.ndarray) -> gpu_sparse_array:
    """Convert numpy array to cupy sparse array

    Parameters
    ----------
    a : numpy.ndarray
        Array to be converted

    Returns
    -------
       gpu_sparse_array : converted array

    """
    return _sparse_to_cupy_sparse(_numpy_to_sparse(a))


def to_arraytype(a: array, atype: str) -> array:
    """Invert arraytype (numpy, scipy.sparse, cupy, cupyx.sparse) to new arraytype

    Parameters
    ----------
    a: array
        Input array to be changed
    atype: str
        Arraytype

    Returns
    -------
        array : Array in specific arraytype

    """
    a, hs_math_current = find_package(a)

    if atype == hs_math_current:
        return a
    else:
        a = eval(
            "_" + short_name_class[hs_math_current] + "_to_" + short_name_class[atype]
        )(a)
        return a


def tonumpy(a: array) -> array:
    return to_arraytype(a, "numpy")


def tosparse(a: array) -> array:
    return to_arraytype(a, "scipy.sparse")


def tocupy(a: array) -> array:
    return to_arraytype(a, "cupy")


def tocupyx(a: array) -> array:
    return to_arraytype(a, "cupyx.sparse")


def soft_threshold(a: array, eps: float) -> array:
    """Soft thresholding of `a` with value of `eps` (clipping is faster than min-max)

    Parameters
    ----------
    a : array
        Input array to be thresholded

    eps : float
        Threshold value

    Returns
    -------
        array: Soft thresholded array

    """
    x = clip(a, -eps, eps)

    return a - x
