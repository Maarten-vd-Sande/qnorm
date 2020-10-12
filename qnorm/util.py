import re
from multiprocessing import Pool, RawArray

import numpy as np


def read_n_lines(file, n):
    """
    Iterate over lines of a file, multiple lines at the same time. This can be
    useful when iterating over multiple files at the same time on a slow
    filesystem (e.g. hard disks). In this case the file can be read on longer
    batches continuously so the reader does not have to switch as often.

    Args:
        file: path to file
        n: number of lines to read at a time

    Returns:
        a list with a string per line
    """
    with open(file) as f:
        lines = []
        for line in f:
            lines.append(line.strip())
            if len(lines) == n:
                yield lines
                lines = []
    yield lines


def _glue_together(lotsalines, delimiter):
    """
    private function of qnorm that that can combine multiple chunks of rows and
    columns into a single table.
    """
    glued = []
    for line in zip(*lotsalines):
        glued.append(delimiter.join(re.split(rf"[\n{delimiter}]+", delimiter.join(line))))
    return "\n".join(glued) + "\n"


def _parallel_argsort(_array, ncpus, dtype):
    """
    private argsort function of qnorm that works with multiple cpus
    """
    # multiproces sorting
    # first we make a shared array
    data_shared = RawArray(
        np.ctypeslib.as_ctypes_type(dtype), _array.shape[0] * _array.shape[1]
    )
    # and wrap it with a numpy array and fill it with our data
    data = np.frombuffer(data_shared, dtype=dtype).reshape(_array.shape)
    np.copyto(data, _array.astype(dtype))

    # now multiprocess sort
    with Pool(
            processes=ncpus,
            initializer=_worker_init,
            initargs=(data_shared, dtype, data.shape),
    ) as pool:
        sorted_idx = np.array(
            pool.map(_worker_sort, range(data.shape[1])), dtype=np.int64
        ).T
    return data, sorted_idx


var_dict = {}


def _worker_init(X, X_dtype, X_shape):
    """
    helper function to pass our reference of X to the sorter
    """
    var_dict["X"] = X
    var_dict["X_dtype"] = X_dtype
    var_dict["X_shape"] = X_shape


def _worker_sort(i):
    """
    argsort a single axis
    """
    X_np = np.frombuffer(var_dict["X"], dtype=var_dict["X_dtype"]).reshape(
        var_dict["X_shape"]
    )
    return np.argsort(X_np[:, i])
