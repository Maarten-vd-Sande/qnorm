import os
import math
from concurrent.futures import ThreadPoolExecutor
from functools import singledispatch
from typing import overload, Union

import numpy as np

from .util import (
    TempFileHolder,
    glue_csv,
    glue_hdf,
    glue_parquet,
    parse_csv,
    parse_hdf,
    parse_parquet,
)

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
                           ncpus: int = 1,
                           ) -> pd.DataFrame: ...

    @overload
    def quantile_normalize(data: np.ndarray,
                           axis: int = 1,
                           target: Union[None, np.ndarray] = None,
                           ncpus: int = 1,
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
            ncpus: number of cpus to use for normalization

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
        values = data.values
        # single precision stays single precision, everything else is
        # normalized in double precision
        if values.dtype != np.float32:
            values = values.astype(np.float64, copy=False)

        qn_values = quantile_normalize_np(values, axis, target, ncpus)
        return pd.DataFrame(qn_values, index=data.index, columns=data.columns)

    def incremental_quantile_normalize(
        infile: str,
        outfile: str,
        rowchunksize: int = 100_000,
        colchunksize: int = 8,
        ncpus: int = 1,
    ) -> None:
        """
        Memory-efficient quantile normalization implementation by splitting
        the task into sequential subtasks, and writing the intermediate results
        to disk instead of keeping them in memory. This makes the memory
        footprint independent of the input table, however also slower..

        Args:
            infile: path to input table. The table can be either a csv-like file
                of which the delimiter is auto detected. Or the infile can be a
                hdf file, which requires to be stored with format=table.
            outfile: path to the output table. Has the same layout and delimiter
                as the input file. If the input is csv-like, the output is csv-
                like. If the input is hdf, then the output is hdf.
            rowchunksize: how many rows to read/write at the same time when
                combining intermediate results. More is faster, but also uses
                more memory.
            colchunksize: how many columns to use at the same time when
                calculating the mean and normalizing. More is faster, but also
                uses more memory.
            ncpus: The number of cpus to use. Scales diminishingly, and more
                than four is generally not useful.
        """
        _check_ncpus(ncpus)
        if infile.endswith((".hdf", ".h5")):
            dataformat = "hdf"
            columns, index = parse_hdf(infile)
        elif infile.endswith((".csv", ".tsv", ".txt")):
            dataformat = "csv"
            columns, index, delimiter = parse_csv(infile)
        elif infile.endswith((".parquet")):
            dataformat = "parquet"
            columns, index, index_used, schema = parse_parquet(infile)
        else:
            raise NotImplementedError(
                "Only HDF ('.hdf', '.h5'), "
                "text ('.csv', '.tsv', '.txt'), "
                "and parquet ('.parquet') formats are supported."
            )

        # now scan the table for which columns and indices it contains
        nr_cols = len(columns)
        nr_rows = len(index)

        # store intermediate tables
        tmp_cols = []

        # calculate the target (rank means)
        target = np.zeros(nr_rows)

        with TempFileHolder() as tfh:
            # loop over our column chunks and keep updating our target
            for i in range(math.ceil(nr_cols / colchunksize)):
                col_start, col_end = (
                    i * colchunksize,
                    np.clip((i + 1) * colchunksize, 0, nr_cols),
                )
                # read relevant columns
                if dataformat == "hdf":
                    with pd.HDFStore(infile) as hdf:
                        assert len(hdf.keys()) == 1
                        key = hdf.keys()[0]
                        cols = [
                            hdf.select_column(key, columns[i])
                            for i in range(col_start, col_end)
                        ]
                        df = pd.concat(cols, axis=1).astype("float32")
                elif dataformat == "csv":
                    df = pd.read_csv(
                        infile,
                        sep=delimiter,
                        comment="#",
                        index_col=0,
                        usecols=[0, *list(range(col_start + 1, col_end + 1))],
                    ).astype("float32")
                elif dataformat == "parquet":
                    df = pd.read_parquet(
                        infile, columns=columns[col_start:col_end]
                    )

                # get the rank means
                if df.values.dtype == np.float32:
                    dtype = np.float32
                else:
                    dtype = np.float64
                data = _to_columns(df.values, dtype, ncpus)
                del df
                rankmeans = _rank_means(data, ncpus)

                # update the target
                target += (rankmeans - target) * (
                    (col_end - col_start) / (col_end)
                )

                # save our intermediate stuff
                tmp_cols.append(
                    tfh.get_filename(prefix="qnorm_", suffix=".npy")
                )
                np.save(tmp_cols[-1], data)
                del data

            # now that we have our target we can start normalizing in chunks
            qnorm_tmp = []

            # store intermediate results
            # and start with our index and store it
            index_tmpfiles = []
            for chunk in np.array_split(
                index, math.ceil(len(index) / rowchunksize)
            ):
                index_tmpfiles.append(
                    tfh.get_filename(prefix="qnorm_", suffix=".p")
                )
                pd.DataFrame(chunk).to_pickle(
                    index_tmpfiles[-1], compression=None
                )
            qnorm_tmp.append(index_tmpfiles)
            del index

            # for each column chunk quantile normalize it onto our distribution
            for i in range(math.ceil(nr_cols / colchunksize)):
                # read the relevant columns in
                data = np.load(tmp_cols[i])

                # quantile normalize
                _normalize_columns(data, target, ncpus)
                qnormed = _to_rows(data, ncpus)
                del data

                # store it in tempfile
                col_tmpfiles = []
                for j, chunk in enumerate(
                    np.array_split(
                        qnormed, math.ceil(qnormed.shape[0] / rowchunksize)
                    )
                ):
                    tmpfile = tfh.get_filename(
                        prefix=f"qnorm_{i}_{j}_", suffix=".npy"
                    )
                    col_tmpfiles.append(tmpfile)
                    np.save(tmpfile, chunk)
                del qnormed, chunk
                qnorm_tmp.append(col_tmpfiles)

            if os.path.exists(outfile):
                os.remove(outfile)

            # glue the separate files together and save them
            if dataformat == "hdf":
                glue_hdf(outfile, columns, qnorm_tmp)
            elif dataformat == "csv":
                glue_csv(outfile, columns, qnorm_tmp, delimiter)
            elif dataformat == "parquet":
                glue_parquet(outfile, columns, qnorm_tmp, index_used, schema)

else:

    @singledispatch
    def quantile_normalize(
        data: np.ndarray,
        axis: int = 1,
        target: Union[None, np.ndarray] = None,
        ncpus: int = 1,
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
            ncpus: number of cpus to use for normalization

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
    # 32 bit data is normalized in single precision, everything else in double
    elif any(
        np.issubdtype(_data.dtype, dtype) for dtype in [np.int32, np.float32]
    ):
        dtype = np.float32
    else:
        dtype = np.float64

    if _data.ndim != 2:
        raise ValueError(
            f"qnorm only supports 2 dimensional data, but your data has "
            f"{_data.ndim} dimensions."
        )

    # take a transposed view of our data if axis is zero
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

    _check_ncpus(ncpus)

    if target is not None:
        # make sure target is correct data type and shape
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
        if target.shape[0] != _data.shape[0]:
            raise ValueError(
                f"The target array does not contain the same amount of values "
                f"({target.shape[0]}) as the data contains rows "
                f"({_data.shape[0]})"
            )
        if not np.issubdtype(target.dtype, np.number):
            raise ValueError(
                f"The type of your target ({target.dtype}) is is not "
                f"supported, and might lead to undefined behaviour. "
                f"Please use numeric data only."
            )
        target = np.sort(target.astype(dtype=dtype)).astype(np.float64)

    # work on a column-major copy, so that every column is contiguous in memory
    data = _to_columns(_data, dtype, ncpus)

    if target is None:
        # if no target supplied get the (sorted) rowmeans
        target = _rank_means(data, ncpus)

    _normalize_columns(data, target, ncpus)

    # return the same memory layout as the input
    if _data.flags.f_contiguous:
        final_res = data.T
    else:
        final_res = _to_rows(data, ncpus)

    if axis == 0:
        final_res = final_res.T
    return final_res


# how many bytes to copy at the same time when changing the memory layout
_BLOCK_BYTES = 2**20
# how many columns to sum together when calculating the target. This is fixed
# so that the order of summation, and thus the result, does not depend on ncpus
_TARGET_COLS = 8


def _check_ncpus(ncpus):
    if not ncpus >= 1:
        raise ValueError("The number of cpus needs to be a positive integer.")


def _parallel(func, tasks, ncpus):
    """
    Lazily map func over tasks with ncpus threads, results are yielded in the
    order of tasks. numpy releases the GIL for sorting, copying and ufuncs, so
    the threads run in parallel.
    """
    if ncpus == 1 or len(tasks) <= 1:
        yield from map(func, tasks)
    else:
        with ThreadPoolExecutor(int(ncpus)) as pool:
            yield from pool.map(func, tasks)


def _row_blocks(n_rows, row_bytes):
    rows = max(1, _BLOCK_BYTES // max(1, row_bytes))
    return [slice(start, start + rows) for start in range(0, n_rows, rows)]


def _to_columns(data, dtype, ncpus):
    """
    Column-major copy of data, so that the columns are contiguous in memory.
    """
    n_rows, n_cols = data.shape
    columns = np.empty((n_cols, n_rows), dtype=dtype)

    def copy(rows):
        columns[:, rows] = data[rows].T

    blocks = _row_blocks(n_rows, n_cols * columns.itemsize)
    for _ in _parallel(copy, blocks, ncpus):
        pass
    return columns


def _to_rows(columns, ncpus):
    """
    Row-major copy of the column-major array columns.
    """
    n_cols, n_rows = columns.shape
    data = np.empty((n_rows, n_cols), dtype=columns.dtype)

    def copy(rows):
        data[rows] = columns[:, rows].T

    blocks = _row_blocks(n_rows, n_cols * columns.itemsize)
    for _ in _parallel(copy, blocks, ncpus):
        pass
    return data


def _rank_means(columns, ncpus):
    """
    The mean of the sorted columns of the column-major array columns.
    """
    n_cols, n_rows = columns.shape

    def sorted_sum(cols):
        total = np.zeros(n_rows)
        sorted_col = np.empty(n_rows, dtype=columns.dtype)
        for col in cols:
            np.copyto(sorted_col, columns[col])
            sorted_col.sort()
            total += sorted_col
        return total

    groups = [
        range(start, min(start + _TARGET_COLS, n_cols))
        for start in range(0, n_cols, _TARGET_COLS)
    ]
    rankmeans = np.zeros(n_rows)
    for total in _parallel(sorted_sum, groups, ncpus):
        rankmeans += total
    return rankmeans / max(n_cols, 1)


def _normalize_columns(columns, target, ncpus):
    """
    Quantile normalize the column-major array columns in-place onto the
    (sorted) target. Tied values all get the mean of the target over the ranks
    they occupy, and NaNs are left as is.
    """
    n_cols, n_rows = columns.shape
    nan = columns.dtype.type(np.nan)

    def normalize(cols):
        sorted_col = np.empty(n_rows, dtype=columns.dtype)
        is_start = np.empty(n_rows, dtype=bool)
        for col in cols:
            column = columns[col]
            sorted_idx = np.argsort(column)
            np.take(column, sorted_idx, out=sorted_col)

            # NaNs are sorted to the end, and are skipped
            n_values = int(np.searchsorted(sorted_col, nan))
            if n_values == 0:
                continue
            values = sorted_col[:n_values]

            # find the blocks of tied values
            starts = is_start[:n_values]
            starts[0] = True
            np.not_equal(values[1:], values[:-1], out=starts[1:])
            if np.count_nonzero(starts) == n_values:
                # no ties, each value simply gets its target value
                np.put(column, sorted_idx[:n_values], target[:n_values])
                continue
            starts = np.flatnonzero(starts)
            counts = np.diff(starts, append=n_values)

            # each value gets the mean of the target over its block
            means = np.add.reduceat(target[:n_values], starts) / counts
            np.put(column, sorted_idx[:n_values], np.repeat(means, counts))

    ncpus = int(min(ncpus, max(n_cols, 1)))
    tasks = [range(start, n_cols, ncpus) for start in range(ncpus)]
    for _ in _parallel(normalize, tasks, ncpus):
        pass
