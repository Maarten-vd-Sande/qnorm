#!/usr/bin/env python
"""Tests for `qnorm` package."""

import unittest
import numpy as np
import pandas as pd

import qnorm


class TestQnorm(unittest.TestCase):
    def test_000_numpy(self):
        """
        test numpy support
        """
        arr = np.random.normal(size=(20, 2))
        qnorm.quantile_normalize(arr)

    def test_001_pandas(self):
        """
        test numpy support
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


if __name__ == "__main__":
    unittest.main()
