from functools import singledispatch
from typing import overload, Union

import numba
import numpy as np

try:
    import pandas as pd

    pandas_import = True
except ModuleNotFoundError:
    pandas_import = False


@numba.jit(nopython=True, fastmath=True, cache=True)
def _quantile_normalize(_in_arr: np.ndarray) -> np.ndarray:
    """
    This function is the heart of this module.

    It does quantile normalization in the "correct" way in the sense that it
    takes the mean of duplicate values instead of ignoring them.
    """
    # get the shape of the input
    n_rows = _in_arr.shape[0]
    n_cols = _in_arr.shape[1]

    # our final output array
    qnorm = np.empty_like(_in_arr, dtype=np.float64)

    # this is quite clunky because numba does not support the axis argument yet
    # sorted- rowmeans, vals, and idx are helper arrays which we need to fill
    # get the value for an index
    sorted_val = np.empty(shape=(n_rows, n_cols), dtype=np.float64)
    # noinspection PyUnresolvedReferences
    sorted_idx = np.empty(
        shape=(n_rows, n_cols), dtype=np.uint32  # type: ignore
    )
    sorted_rowmeans = np.empty(shape=n_rows, dtype=np.float64)
    for col_i in range(n_cols):
        argsort = np.argsort(_in_arr[:, col_i])
        sorted_idx[:, col_i] = argsort
        sorted_val[:, col_i] = np.array([_in_arr[i, col_i] for i in argsort])

    for row in range(n_rows):
        sorted_rowmeans[row] = np.mean(sorted_val[row, :])

    # we quantile normalize separately per column
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
                val += sorted_rowmeans[i + n]
                n += 1

            # fill out qnorm with our new value
            if n > 0:
                val /= n
                for j in range(n):
                    idx = sorted_idx[i + j, col_i]
                    qnorm[idx, col_i] = val

            i += n

    return qnorm


if pandas_import:
    # fmt: off
    # function overloading for the correct return type depending on the input
    @overload
    def quantile_normalize(data: pd.DataFrame) -> pd.DataFrame: ...
    @overload
    def quantile_normalize(data: np.ndarray) -> np.ndarray: ...
    # fmt: on

    @singledispatch
    def quantile_normalize(
        data: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Quantile normalize your array/dataframe.

        returns: a quantile normalized copy of the input.
        """
        raise NotImplementedError(
            f"quantile_normalize not implemented for type {type(data)}"
        )

    @quantile_normalize.register(pd.DataFrame)
    def quantile_normalize_pd(data: pd.DataFrame) -> pd.DataFrame:
        qn_data = data.copy()
        qn_data[:] = quantile_normalize_np(qn_data.values)
        return qn_data


else:

    @singledispatch
    def quantile_normalize(data: np.ndarray) -> np.ndarray:
        """
        Quantile normalize your array.

        returns: a quantile normalized copy of the input.
        """
        raise NotImplementedError(
            f"quantile_normalize not implemented for type {type(data)}"
        )


@quantile_normalize.register(np.ndarray)
def quantile_normalize_np(data: np.ndarray) -> np.ndarray:
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(
            f"The type of your data ({data.dtype}) is is not "
            f"supported, and might lead to undefined behaviour. "
            f"Please use numeric data only."
        )

    data = data.astype(dtype=np.float64)
    return _quantile_normalize(data)
