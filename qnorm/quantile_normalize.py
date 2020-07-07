"""Main module."""
import numpy as np
import numba


# @numba.jit(nopython=True, fastmath=True, cache=True)
def quantile_normalize(_in_arr):
    """
    Quantile normalization
    """
    n_rows = _in_arr.shape[0]
    n_cols = _in_arr.shape[1]
    qnorm = np.empty_like(_in_arr, dtype=np.float64)

    sorted_val = np.empty(shape=(n_rows, n_cols), dtype=np.float64)
    sorted_idx = np.empty(shape=(n_rows, n_cols), dtype=np.uint32)
    sorted_rowmeans = np.empty(shape=n_rows, dtype=np.float64)
    for col_i in range(n_cols):
        argsort = np.argsort(_in_arr[:, col_i])
        sorted_idx[:, col_i] = argsort
        sorted_val[:, col_i] = np.array([_in_arr[i, col_i] for i in argsort])

    for row in range(n_rows):
        sorted_rowmeans[row] = np.mean(sorted_val[row])

    for col_i in range(n_cols):
        i = 0
        while i < n_rows:
            n = val = 0
            while i + n < n_rows and sorted_val[i, col_i] == sorted_val[i + n, col_i]:
                val += sorted_rowmeans[i + n]
                n += 1

            if n > 0:
                val /= n
                for j in range(n):
                    idx = sorted_idx[i + j, col_i]
                    qnorm[idx, col_i] = val

            i += n

    return qnorm
