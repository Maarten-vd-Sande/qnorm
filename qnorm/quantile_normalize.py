import os
import math
from functools import singledispatch
from typing import overload, Union

import numba
import numpy as np

from .util import (
    TempFileHolder,
    glue_csv,
    glue_hdf,
    parse_csv,
    parse_hdf,
    _parallel_argsort,
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
    def quantile_normalize(data: str,
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
        qn_data = data.copy()

        # if we use axis 0, then already transpose here, and not later
        if axis == 0:
            qn_data = qn_data.T
            axis = 1

        qn_data[:] = quantile_normalize_np(qn_data.values, axis, target, ncpus)
        return qn_data

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
        if infile.endswith((".hdf", ".h5")):
            dataformat = "hdf"
            columns, index = parse_hdf(infile)
        elif infile.endswith((".csv", ".tsv", ".txt")):
            dataformat = "csv"
            columns, index, delimiter = parse_csv(infile)
        else:
            raise NotImplementedError("")

        # now scan the table for which columns and indices it contains
        nr_cols = len(columns)
        nr_rows = len(index)

        # store intermediate tables
        tmp_vals = []
        tmp_sorted_vals = []
        tmp_idxs = []

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

                # get the rank means
                data, sorted_idx = _parallel_argsort(
                    df.values, ncpus, df.values.dtype
                )
                del df
                sorted_vals = np.take_along_axis(
                    data,
                    sorted_idx,
                    axis=0,
                )
                rankmeans = np.mean(sorted_vals, axis=1)

                # update the target
                target += (rankmeans - target) * (
                    (col_end - col_start) / (col_end)
                )

                # save all our intermediate stuff
                tmp_vals.append(
                    tfh.get_filename(prefix="qnorm_", suffix=".npy")
                )
                tmp_sorted_vals.append(
                    tfh.get_filename(prefix="qnorm_", suffix=".npy")
                )
                tmp_idxs.append(
                    tfh.get_filename(prefix="qnorm_", suffix=".npy")
                )
                np.save(tmp_vals[-1], data)
                np.save(tmp_sorted_vals[-1], sorted_vals)
                np.save(tmp_idxs[-1], sorted_idx)
                del data, sorted_idx, sorted_vals

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
                data = np.load(tmp_vals[i], allow_pickle=True)
                sorted_idx = np.load(tmp_idxs[i], allow_pickle=True)
                sorted_vals = np.load(tmp_sorted_vals[i], allow_pickle=True)

                # quantile normalize
                qnormed = _numba_accel_qnorm(
                    data, sorted_idx, sorted_vals, target
                )
                del data, sorted_idx, sorted_vals

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
        data, sorted_idx = _parallel_argsort(_data, ncpus, dtype)
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
        target = np.sort(target.astype(dtype=dtype))

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
