#!/usr/bin/env python
"""Tests for `qnorm` package."""

import unittest
import numpy as np
import pandas as pd

import qnorm
import tracemalloc


tracemalloc.start()

df1 = pd.DataFrame(
    {
        "C1": {"A": 5.0, "B": 2.0, "C": 3.0, "D": 4.0},
        "C2": {"A": 4.0, "B": 1.0, "C": 4.0, "D": 2.0},
        "C3": {"A": 3.0, "B": 4.0, "C": 6.0, "D": 8.0},
    }
)
df1.to_csv("test.csv")
df1.to_hdf("test.hdf", key="qnorm", format="table", data_columns=True, mode="w")


class TestQnorm(unittest.TestCase):
    def test_000_numpy(self):
        """
        test numpy support
        """
        arr = np.random.normal(size=(20, 2))
        qnorm.quantile_normalize(arr)

    def test_001_pandas(self):
        """
        test pandas support
        """
        df = pd.DataFrame(
            {
                "C1": {"A": 5.0, "B": 2.0, "C": 3.0, "D": 4.0},
                "C2": {"A": 4.0, "B": 1.0, "C": 4.0, "D": 2.0},
                "C3": {"A": 3.0, "B": 4.0, "C": 6.0, "D": 8.0},
            }
        )
        qnorm.quantile_normalize(df)

    def test_002_wiki(self):
        """
        test the wiki example
        https://en.wikipedia.org/wiki/Quantile_normalization
        """
        df = pd.DataFrame(
            {
                "C1": {"A": 5.0, "B": 2.0, "C": 3.0, "D": 4.0},
                "C2": {"A": 4.0, "B": 1.0, "C": 4.0, "D": 2.0},
                "C3": {"A": 3.0, "B": 4.0, "C": 6.0, "D": 8.0},
            }
        )

        result = np.array(
            [
                [5.66666667, 5.16666667, 2.0],
                [2.0, 2.0, 3.0],
                [3.0, 5.16666667, 4.66666667],
                [4.66666667, 3.0, 5.66666667],
            ]
        )

        np.testing.assert_array_almost_equal(
            qnorm.quantile_normalize(df).values, result
        )

    def test_003_no_change(self):
        """
        no sorting should happen here
        """
        arr = np.empty(shape=(20, 3))
        for col in range(arr.shape[1]):
            vals = np.arange(arr.shape[0])
            np.random.shuffle(vals)
            arr[:, col] = vals

        qnorm_arr = qnorm.quantile_normalize(arr)
        np.testing.assert_array_almost_equal(arr, qnorm_arr)

    def test_004_double(self):
        """
        if dtype is double, return double
        """
        arr = np.random.normal(0, 1, size=(20, 3))
        arr = arr.astype(np.float64)
        qnorm_arr = qnorm.quantile_normalize(arr)
        assert qnorm_arr.dtype == np.float64

    def test_005_single(self):
        """
        if dtype is single, return single
        """
        arr = np.random.normal(0, 1, size=(20, 3))
        arr = arr.astype(np.float32)
        qnorm_arr = qnorm.quantile_normalize(arr)
        assert qnorm_arr.dtype == np.float32

    def test_006_target(self):
        """
        test if the target is used instead of the qnorm values
        """
        arr = np.array([np.arange(0, 10), np.arange(0, 10)]).T
        np.random.shuffle(arr)
        target = np.arange(10, 20)
        qnorm_arr = qnorm.quantile_normalize(arr, target=target)
        for val in target:
            assert (
                val in qnorm_arr[:, 0] and val in qnorm_arr[:, 1]
            ), f"value {val} not in qnorm array"

    def test_007_target_notsorted(self):
        """
        make sure an unsorted target gets sorted first
        """
        arr = np.array([np.arange(0, 10), np.arange(0, 10)]).T
        np.random.shuffle(arr)
        # take the reverse, which should be sorted by qnorm
        target = np.arange(10, 20)[::-1]
        qnorm_arr = qnorm.quantile_normalize(arr, target=target)
        for val in target:
            assert (
                val in qnorm_arr[:, 0] and val in qnorm_arr[:, 1]
            ), f"value {val} not in qnorm array"

    def test_008_short_target(self):
        """
        test if an error is raised with a invalid sized target
        """
        arr = np.array([np.arange(0, 10), np.arange(0, 10)]).T
        target = np.arange(10, 15)
        self.assertRaises(ValueError, qnorm.quantile_normalize, arr, target)

    def test_009_wiki_ncpus(self):
        """
        test if an error is raised with a invalid sized target
        """
        df = pd.DataFrame(
            {
                "C1": {"A": 5.0, "B": 2.0, "C": 3.0, "D": 4.0},
                "C2": {"A": 4.0, "B": 1.0, "C": 4.0, "D": 2.0},
                "C3": {"A": 3.0, "B": 4.0, "C": 6.0, "D": 8.0},
            }
        )

        result = np.array(
            [
                [5.66666667, 5.16666667, 2.0],
                [2.0, 2.0, 3.0],
                [3.0, 5.16666667, 4.66666667],
                [4.66666667, 3.0, 5.66666667],
            ]
        )

        np.testing.assert_array_almost_equal(
            qnorm.quantile_normalize(df, ncpus=10).values, result
        )

    def test_010_axis_numpy(self):
        """
        test numpy axis support
        """
        arr = np.random.normal(size=(50, 4))

        np.testing.assert_array_almost_equal(
            qnorm.quantile_normalize(arr.T, axis=0),
            qnorm.quantile_normalize(arr, axis=1),
        )
        np.testing.assert_array_almost_equal(
            qnorm.quantile_normalize(arr, axis=1),
            qnorm.quantile_normalize(arr.T, axis=0),
        )

    def test_011_axis_pandas(self):
        """
        test numpy axis support
        """
        df = pd.DataFrame(
            {
                "C1": {"A": 5.0, "B": 2.0, "C": 3.0, "D": 4.0},
                "C2": {"A": 4.0, "B": 1.0, "C": 4.0, "D": 2.0},
                "C3": {"A": 3.0, "B": 4.0, "C": 6.0, "D": 8.0},
            }
        )

        np.testing.assert_array_almost_equal(
            qnorm.quantile_normalize(df.T, axis=0),
            qnorm.quantile_normalize(df, axis=1),
        )
        np.testing.assert_array_almost_equal(
            qnorm.quantile_normalize(df, axis=1),
            qnorm.quantile_normalize(df.T, axis=0),
        )

    def test_012_from_csv(self):
        """
        test the basic incremental_quantile_normalize functionality
        """
        qnorm.incremental_quantile_normalize("test.csv", "test_out.csv")
        df1 = pd.read_csv("test.csv", index_col=0, header=0)
        df2 = pd.read_csv("test_out.csv", index_col=0, header=0)

        np.testing.assert_almost_equal(
            qnorm.quantile_normalize(df1), df2.values, decimal=5
        )

    def test_013_from_csv_rowchunk(self):
        """
        test the incremental_quantile_normalize with rowchunks functionality
        """
        df1 = pd.read_csv("test.csv", index_col=0, header=0)

        for rowchunksize in range(1, 10):
            qnorm.incremental_quantile_normalize(
                "test.csv", "test_out.csv", rowchunksize=rowchunksize
            )
            df2 = pd.read_csv("test_out.csv", index_col=0, header=0)

            np.testing.assert_almost_equal(
                qnorm.quantile_normalize(df1), df2.values, decimal=5
            )

    def test_014_from_csv_colchunk(self):
        """
        test the incremental_quantile_normalize with colchunks functionality
        """
        df1 = pd.read_csv("test.csv", index_col=0, header=0)

        for colchunksize in range(1, 10):
            qnorm.incremental_quantile_normalize(
                "test.csv", "test_out.csv", colchunksize=colchunksize
            )
            df2 = pd.read_csv("test_out.csv", index_col=0, header=0)

            np.testing.assert_almost_equal(
                qnorm.quantile_normalize(df1), df2.values, decimal=5
            )

    def test_015_from_csv_colrowchunk(self):
        """
        test the incremental_quantile_normalize with both row and colchunks
        """
        df1 = pd.read_csv("test.csv", index_col=0, header=0)

        for colchunksize in range(1, 10):
            for rowchunksize in range(1, 10):
                qnorm.incremental_quantile_normalize(
                    "test.csv",
                    "test_out.csv",
                    rowchunksize=rowchunksize,
                    colchunksize=colchunksize,
                )
                df2 = pd.read_csv("test_out.csv", index_col=0, header=0)

                np.testing.assert_almost_equal(
                    qnorm.quantile_normalize(df1), df2.values, decimal=5
                )

    def test_016_from_csv_largefile(self):
        """
        test whether or not incremental_quantile_normalize works with a larger random
        file
        """
        np.random.seed(42)
        df1 = pd.DataFrame(index=range(5000), columns=range(100))
        df1[:] = np.random.randint(0, 100, size=df1.shape)
        df1.to_csv("test_large.csv")

        qnorm.incremental_quantile_normalize(
            "test_large.csv",
            "test_large_out.csv",
            rowchunksize=11,
            colchunksize=11,
        )
        df2 = pd.read_csv("test_large_out.csv", index_col=0, header=0)

        np.testing.assert_almost_equal(
            qnorm.quantile_normalize(df1), df2.values, decimal=4
        )

    def test_017_from_hdf(self):
        """
        test the basic incremental_quantile_normalize functionality
        """
        qnorm.incremental_quantile_normalize("test.hdf", "test_out.hdf")
        df1 = pd.read_hdf("test.hdf", index_col=0, header=0)
        df2 = pd.read_hdf("test_out.hdf", index_col=0, header=0)

        np.testing.assert_almost_equal(
            qnorm.quantile_normalize(df1), df2.values, decimal=5
        )

    def test_018_from_hdf_rowchunk(self):
        """
        test the incremental_quantile_normalize with rowchunks functionality
        """
        df1 = pd.read_hdf("test.hdf", index_col=0, header=0)

        for rowchunksize in range(1, 10):
            qnorm.incremental_quantile_normalize(
                "test.hdf", "test_out.hdf", rowchunksize=rowchunksize
            )
            df2 = pd.read_hdf("test_out.hdf", index_col=0, header=0)

            np.testing.assert_almost_equal(
                qnorm.quantile_normalize(df1), df2.values, decimal=5
            )

    def test_019_from_hdf_colchunk(self):
        """
        test the incremental_quantile_normalize with colchunks functionality
        """
        df1 = pd.read_hdf("test.hdf", index_col=0, header=0)

        for colchunksize in range(1, 10):
            qnorm.incremental_quantile_normalize(
                "test.hdf", "test_out.hdf", colchunksize=colchunksize
            )
            df2 = pd.read_hdf("test_out.hdf", index_col=0, header=0)

            np.testing.assert_almost_equal(
                qnorm.quantile_normalize(df1), df2.values, decimal=5
            )

    def test_020_from_hdf_colrowchunk(self):
        """
        test the incremental_quantile_normalize with both row and colchunks
        """
        df1 = pd.read_hdf("test.hdf", index_col=0, header=0)

        for colchunksize in range(1, 10):
            for rowchunksize in range(1, 10):
                qnorm.incremental_quantile_normalize(
                    "test.hdf",
                    "test_out.hdf",
                    rowchunksize=rowchunksize,
                    colchunksize=colchunksize,
                )
                df2 = pd.read_hdf("test_out.hdf", index_col=0, header=0)

                np.testing.assert_almost_equal(
                    qnorm.quantile_normalize(df1), df2.values, decimal=5
                )

    def test_021_from_hdf_largefile(self):
        """
        test whether or not incremental_quantile_normalize works with a larger random
        file
        """
        np.random.seed(42)
        df1 = pd.DataFrame(
            index=range(5000),
            columns=["sample" + str(col) for col in range(100)],
        )
        df1[:] = np.random.randint(0, 100, size=df1.shape)
        df1.to_hdf(
            "test_large.hdf", key="qnorm", format="table", data_columns=True
        )

        qnorm.incremental_quantile_normalize(
            "test_large.hdf",
            "test_large_out.hdf",
            rowchunksize=11,
            colchunksize=11,
        )
        df2 = pd.read_hdf("test_large_out.hdf", index_col=0, header=0)

        np.testing.assert_almost_equal(
            qnorm.quantile_normalize(df1), df2.values, decimal=4
        )

    def test_022(self):
        """
        Test another array, not just wiki example.
        """
        df = pd.DataFrame(
            {
                "C1": {
                    "A": 2.0,
                    "B": 2.0,
                    "C": 2.0,
                    "D": 2.0,
                    "E": 6.0,
                    "F": 1.0,
                },
                "C2": {
                    "A": 2.0,
                    "B": 2.0,
                    "C": 1.0,
                    "D": 3.5,
                    "E": 5.0,
                    "F": 1.0,
                },
            }
        )
        np.testing.assert_almost_equal(
            qnorm.quantile_normalize(df).values,
            np.array(
                [
                    [2.0625, 2.0],
                    [2.0625, 2.0],
                    [2.0625, 1.25],
                    [2.0625, 2.75],
                    [5.5, 5.5],
                    [1.0, 1.25],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
