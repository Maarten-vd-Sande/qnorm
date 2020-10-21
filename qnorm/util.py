import os
import warnings
import tempfile
import random
import string
from pathlib import Path
from multiprocessing import Pool, RawArray

import numpy as np


class TempFileHolder:
    def __enter__(self):
        self.tmpfiles = list()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # remove all the files
        for file in self.tmpfiles:
            if os.path.isfile(file):
                os.remove(file)

    def get_filename(self, prefix="", suffix=""):
        tmpdir = tempfile.gettempdir()
        for i in range(100):
            rand_seq = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=8)
            )
            filename = f"{tmpdir}/{prefix}{rand_seq}{suffix}"
            if os.path.exists(filename):
                continue
            else:
                Path(filename).touch()
                self.tmpfiles.append(filename)
                return filename


def parse_csv(infile):
    """
    parse a csv file (memory efficient) and get the columns, index and
    delimiter from it.
    """
    import pandas as pd

    delimiter = get_delim(infile)
    columns = [
        str(col)
        for col in pd.read_csv(
            infile, sep=delimiter, nrows=10, comment="#", index_col=0
        ).columns
    ]
    index = [
        str(row)
        for row in pd.read_csv(
            infile, sep=delimiter, comment="#", index_col=0, usecols=[0]
        ).index
    ]
    return columns, index, delimiter


def parse_hdf(infile):
    """
    parse a hdf file and get the columns and index from it.
    """
    import pandas as pd

    # TODO: only table format
    columns = [col for col in pd.read_hdf(infile, start=0, stop=0).columns]
    with pd.HDFStore(infile) as hdf:
        assert len(hdf.keys()) == 1
        key = hdf.keys()[0]
        index = hdf.select_column(key, "index").values
    return columns, index


def glue_csv(outfile, header, colfiles, delimiter):
    """
    glue multiple csv into a single csv
    """
    open_colfiles = [read_lines(tmpfiles) for tmpfiles in colfiles]

    # now collapse everything together
    with open(outfile, "w") as outfile:
        # add our columns/header section
        outfile.write(delimiter.join([""] + header) + "\n")

        # now start reading our chunked columns and chunked rows and write them
        for lotsalines in zip(*open_colfiles):
            outfile.write(_glue_csv(lotsalines, delimiter))


def _glue_csv(lotsalines, delimiter):
    """
    private function of qnorm that that can combine multiple chunks of rows and
    columns into a single table.
    """
    stack = np.hstack(lotsalines)
    fmt = delimiter.join(["%s"] + ["%g"] * (stack.shape[1] - 1))
    fmt = "\n".join([fmt] * stack.shape[0])
    data = fmt % tuple(stack.ravel())
    return data + "\n"


def glue_hdf(outfile, header, colfiles):
    """
    glue multiple hdf into a single hdf
    """
    import pandas as pd

    open_colfiles = [read_lines(tmpfiles) for tmpfiles in colfiles]

    for lotsalines in zip(*open_colfiles):
        df = pd.DataFrame(np.hstack(lotsalines))
        df.set_index(0, inplace=True)
        df.index.name = None
        df.columns = header
        df = df.astype("float32")
        df.to_hdf(
            outfile,
            key="qnorm",
            append=True,
            mode="a",
            format="table",
            min_itemsize=15,
        )


def get_delim(table):
    import pandas as pd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        delimiter = pd.read_csv(
            table, sep=None, iterator=True, nrows=1000, comment="#", index_col=0
        )._engine.data.dialect.delimiter
    return delimiter


def read_lines(files):
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
    for file in files:
        yield np.load(file, allow_pickle=True)


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
