from multiprocessing import Pool, RawArray
from functools import singledispatch
from typing import overload, Union

import numba
import numpy as np

from .util import worker_init, worker_sort

try:
    import pandas as pd

    pandas_import = True
except ModuleNotFoundError:
    pandas_import = False


if pandas_import:
    # fmt: off
    # function overloading for the correct return type depending on the input
    @overload
    def quantile_normalize(data: pd.DataFrame,
                           axis: int = 1,
                           target: Union[None, np.ndarray] = None,
                           ncpus: int = 1
                           ) -> pd.DataFrame: ...

    @overload
    def quantile_normalize(data: np.ndarray,
                           axis: int = 1,
                           target: Union[None, np.ndarray] = None,
                           ncpus: int = 1
                           ) -> np.ndarray: ...
    # fmt: on

    @singledispatch
    def quantile_normalize(
        data: Union[pd.DataFrame, np.ndarray],
        axis: int = 1,
        target: Union[None, np.ndarray] = None,
        ncpus: int = 1,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Quantile normalize your array/dataframe.

        It does quantile normalization in the "correct" way in the sense that
        it takes the mean of duplicate values instead of ignoring them.

        Args:
            data: numpy.ndarray or pandas.DataFrame to be normalized
            axis: axis along to normalize. Axis=1 (default) normalizes each
                  column/sample which gives them identical distributions.
                  Axis=0 normalizes each row/feature giving them all identical
                  distributions.
            target: distribution to normalize onto

        Returns: a quantile normalized copy of the input.
        """
        raise NotImplementedError(
            f"quantile_normalize not implemented for type {type(data)}"
        )

    @quantile_normalize.register(pd.DataFrame)
    def quantile_normalize_pd(
        data: pd.DataFrame,
        axis: int = 1,
        target: Union[None, np.ndarray] = None,
        ncpus: int = 1,
    ) -> pd.DataFrame:
        qn_data = data.copy()

        # if we use axis 0, then already transpose here, and not later
        if axis == 0:
            qn_data = qn_data.T
            axis = 1

        qn_data[:] = quantile_normalize_np(qn_data.values, axis, target, ncpus)
        return qn_data


else:

    @singledispatch
    def quantile_normalize(
        data: np.ndarray, axis: int = 1, target: Union[None, np.ndarray] = None
    ) -> np.ndarray:
        """
        Quantile normalize your array.

        It does quantile normalization in the "correct" way in the sense that
        it takes the mean of duplicate values instead of ignoring them.

        Args:
            data: numpy.ndarray or pandas.DataFrame to be normalized
            axis: axis along to normalize. Axis=1 (default) normalizes each
                  column/sample which gives them identical distributions.
                  Axis=0 normalizes each row/feature giving them all identical
                  distributions.
            target: distribution to normalize onto

        Returns: a quantile normalized copy of the input.
        """
        raise NotImplementedError(
            f"quantile_normalize not implemented for type {type(data)}"
        )


@quantile_normalize.register(np.ndarray)
def quantile_normalize_np(
    _data: np.ndarray,
    axis: int = 1,
    target: Union[None, np.ndarray] = None,
    ncpus: int = 1,
) -> np.ndarray:
    # check for supported dtypes
    if not np.issubdtype(_data.dtype, np.number):
        raise ValueError(
            f"The type of your data ({_data.dtype}) is is not "
            f"supported, and might lead to undefined behaviour. "
            f"Please use numeric data only."
        )
    # numba does not (yet) support smaller
    elif any(
        np.issubdtype(_data.dtype, dtype) for dtype in [np.int32, np.float32]
    ):
        dtype = np.float32
    else:
        dtype = np.float64

    # take a transposed view of our data if axis is one
    if axis == 0:
        _data = np.transpose(_data)
    elif axis == 1:
        pass
    else:
        raise ValueError(
            f"qnorm only supports 2 dimensional data, so the axis"
            f"has to be either 0 or 1, but you set axis to "
            f"{axis}."
        )

    # sort the array, single process or multiprocessing
    if ncpus == 1:
        # single process sorting
        data = _data.astype(dtype=dtype)
        # we do the sorting outside of numba because the numpy implementation
        # is faster, and numba does not support the axis argument.
        sorted_idx = np.argsort(data, axis=0)
    elif ncpus > 1:
        # multiproces sorting
        # first we make a shared array
        data_shared = RawArray(
            np.ctypeslib.as_ctypes_type(dtype), _data.shape[0] * _data.shape[1]
        )
        # and wrap it with a numpy array and fill it with our data
        data = np.frombuffer(data_shared, dtype=dtype).reshape(_data.shape)
        np.copyto(data, _data.astype(dtype))

        # now multiprocess sort
        with Pool(
            processes=ncpus,
            initializer=worker_init,
            initargs=(data_shared, dtype, data.shape),
        ) as pool:
            sorted_idx = np.array(
                pool.map(worker_sort, range(data.shape[1])), dtype=np.int64
            ).T
    else:
        raise ValueError("The number of cpus needs to be a positive integer.")

    sorted_val = np.take_along_axis(data, sorted_idx, axis=0)

    if target is None:
        # if no target supplied get the (sorted) rowmeans
        target = np.mean(sorted_val, axis=1)
    else:
        # otherwise make sure target is correct data type and shape
        if not isinstance(target, np.ndarray):
            try:
                target = np.array(target)
            except Exception:
                raise ValueError(
                    "The target could not be converted to a " "numpy.ndarray."
                )
        if target.ndim != 1:
            raise ValueError(
                f"The target array should be a 1-dimensionsal vector, however "
                f"you supplied a vector with {target.ndim} dimensions"
            )
        if target.shape[0] != data.shape[0]:
            raise ValueError(
                f"The target array does not contain the same amount of values "
                f"({target.shape[0]}) as the data contains rows "
                f"({data.shape[0]})"
            )
        if not np.issubdtype(target.dtype, np.number):
            raise ValueError(
                f"The type of your target ({data.dtype}) is is not "
                f"supported, and might lead to undefined behaviour. "
                f"Please use numeric data only."
            )
        target = target.astype(dtype=dtype)

    return _numba_accel_qnorm(data, sorted_idx, sorted_val, target)


@numba.jit(nopython=True, fastmath=True, cache=True)
def _numba_accel_qnorm(
    qnorm: np.ndarray,
    sorted_idx: np.ndarray,
    sorted_val: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    numba accelerated "actual" qnorm normalization.
    """
    # get the shape of the input
    n_rows = qnorm.shape[0]
    n_cols = qnorm.shape[1]

    for col_i in range(n_cols):
        i = 0
        # we fill out a column not from lowest index to highest index,
        # but we fill out qnorm from lowest value to highest value
        while i < n_rows:
            n = 0
            val = 0.0
            # since there might be duplicate numbers in a column, we search for
            # all the indices that have these duplcate numbers. Then we take
            # the mean of their rowmeans.
            while (
                i + n < n_rows
                and sorted_val[i, col_i] == sorted_val[i + n, col_i]
            ):
                val += target[i + n]
                n += 1

            # fill out qnorm with our new value
            if n > 0:
                val /= n
                for j in range(n):
                    idx = sorted_idx[i + j, col_i]
                    qnorm[idx, col_i] = val

            i += n

    return qnorm
