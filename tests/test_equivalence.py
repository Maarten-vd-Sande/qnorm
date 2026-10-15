"""
Behaviour-lock tests for qnorm.

These tests compare qnorm against a deliberately naive reference
implementation, and against golden outputs that were generated with an
earlier implementation of qnorm (see tests/make_golden.py).
"""

import importlib
import math
import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import qnorm

# qnorm.quantile_normalize is the function, this is the module
QN_MODULE = importlib.import_module("qnorm.quantile_normalize")
GOLDEN = os.path.join(os.path.dirname(__file__), "data", "golden.npz")

SHAPES = [(1, 1), (1, 5), (5, 1), (7, 3), (1000, 17), (20000, 4)]
DTYPES = [np.int32, np.int64, np.uint8, np.float32, np.float64]
KINDS = ["unique", "ties", "counts", "mixed", "constant"]
NANS = [0.0, 0.1, "allnan"]
LAYOUTS = ["C", "F", "strided"]


def _qnorm_dtype(dtype):
    """the dtype qnorm calculates and returns in"""
    if np.dtype(dtype) in (np.dtype(np.int32), np.dtype(np.float32)):
        return np.float32
    return np.float64


def _reference_qnorm(data, target=None):
    """
    Slow but obviously correct quantile normalization along axis=1.

    Every value is replaced by the mean of the target over the ranks that its
    tie block occupies. The ranks of the tie block of value v are found with
    searchsorted on the sorted column: [#values < v, #values <= v). NaNs are
    left in place and sort to the end.
    """
    dtype = _qnorm_dtype(data.dtype)
    data = np.asarray(data).astype(dtype)
    n_rows, n_cols = data.shape
    sorted_cols = np.sort(data, axis=0)

    if target is None:
        target = sorted_cols.astype(np.float64).mean(axis=1)
    else:
        target = np.sort(np.asarray(target).astype(dtype)).astype(np.float64)

    out = np.full(data.shape, np.nan, dtype=dtype)
    for col in range(n_cols):
        values = data[:, col]
        los = np.searchsorted(sorted_cols[:, col], values, side="left")
        his = np.searchsorted(sorted_cols[:, col], values, side="right")
        block_means = {}
        for row in range(n_rows):
            if np.isnan(values[row]):
                continue
            lo, hi = los[row], his[row]
            if (lo, hi) not in block_means:
                block_means[(lo, hi)] = target[lo:hi].mean()
            out[row, col] = block_means[(lo, hi)]
    return out


def _make_values(rng, kind, shape):
    n_rows, n_cols = shape
    if kind == "unique":
        return rng.standard_normal(shape) * 1000
    if kind == "ties":
        return rng.integers(0, 4, shape).astype(float)
    if kind == "counts":
        return rng.negative_binomial(2, 0.01, shape).astype(float)
    if kind == "constant":
        return np.full(shape, 7.0)
    # mixed: every column a different kind
    kinds = ["unique", "ties", "counts", "constant"]
    columns = [
        _make_values(rng, kinds[col % len(kinds)], (n_rows, 1))
        for col in range(n_cols)
    ]
    return np.hstack(columns)


def _make_array(rng, kind, shape, dtype, nans=0.0):
    values = _make_values(rng, kind, shape)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = np.clip(np.round(values), info.min, info.max)
    arr = values.astype(dtype)
    if np.issubdtype(dtype, np.floating):
        if nans == "allnan":
            arr[:, 0] = np.nan
        elif nans > 0:
            arr[rng.random(shape) < nans] = np.nan
    return arr


def _with_layout(arr, layout):
    if layout == "C":
        return np.ascontiguousarray(arr)
    if layout == "F":
        return np.asfortranarray(arr)
    # non-contiguous view with a negative stride
    big = np.zeros((arr.shape[0] * 2, arr.shape[1]), dtype=arr.dtype)
    big[::2, ::-1] = arr
    return big[::2, ::-1]


def _assert_close(test, actual, expected, msg=""):
    test.assertEqual(actual.shape, expected.shape, msg)
    test.assertEqual(actual.dtype, expected.dtype, msg)
    if expected.dtype == np.float32:
        rtol = 1e-6
    else:
        rtol = 1e-12
    finite = expected[np.isfinite(expected)]
    scale = max(1.0, float(np.abs(finite).max())) if finite.size else 1.0
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=rtol * scale, err_msg=msg
    )


def _extreme_array(rng, kind, shape, dtype):
    """
    values at the edges of the floating point range, with random signs:
    huge: close to the maximum (for float64 small enough for the sum to fit)
    tiny: subnormals
    range: magnitudes spread over the whole exponent range
    cancel: columns alternate between around +1e30 and -1e30, so that the
        values of a rank (almost) cancel each other out
    """
    exponents = {
        np.float32: {
            "huge": (36, 38.5),
            "tiny": (-44.5, -37),
            "range": (-44, 38),
        },
        np.float64: {
            "huge": (304, 306),
            "tiny": (-323, -308),
            "range": (-323, 306),
        },
    }
    if kind == "cancel":
        spread = 1e-4 if dtype == np.float32 else 1e-10
        values = 1e30 * (1 + spread * rng.random(shape))
        values[:, 1::2] *= -1
    else:
        low, high = exponents[dtype][kind]
        values = 10 ** rng.uniform(low, high, shape)
        values *= rng.choice([-1, 1], shape)
    return values.astype(dtype)


def _exact_qnorm(data):
    """
    Quantile normalization of float data along axis=1 with exactly rounded
    rank means (math.fsum), plus per value how far off a correct
    implementation can be: floating point summation in any order of the n
    values of a rank is off by at most (n - 1) * eps/2 * sum(|values|), after
    which the tie blocks are averaged and the result is rounded to the dtype.
    """
    n_rows, n_cols = data.shape
    eps = np.finfo(np.float64).eps
    sorted_cols = np.sort(data, axis=0)
    ranks = sorted_cols.astype(np.float64)
    target = np.array([math.fsum(rank) for rank in ranks]) / n_cols
    target_tol = eps * (np.abs(ranks).sum(axis=1) + np.abs(target))

    out = np.empty(data.shape)
    tol = np.empty(data.shape)
    for col in range(n_cols):
        values = data[:, col]
        los = np.searchsorted(sorted_cols[:, col], values, side="left")
        his = np.searchsorted(sorted_cols[:, col], values, side="right")
        for row in range(n_rows):
            lo, hi = los[row], his[row]
            block = target[lo:hi]
            out[row, col] = math.fsum(block) / len(block)
            block_tol = (len(block) + 1) * eps * np.abs(block).max()
            tol[row, col] = target_tol[lo:hi].max() + block_tol
    info = np.finfo(data.dtype)
    tol += info.eps * np.abs(out) + info.smallest_subnormal
    return out, tol


def _special_array(dtype):
    """array with infinities, signed zeros, NaNs, denormals and big values"""
    col0 = [-np.inf, -0.0, 0.0, 1.0, 1.0, np.inf, np.nan, 3.0]
    col1 = [0.0, -0.0, -0.0, np.nan, np.nan, 2.0, -1.0, 5.0]
    col2 = [1e-40, -1e-40, 0.0, 1e30, -1e30, 1e30, 2.0, 2.0]
    col3 = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]
    return np.array([col0, col1, col2, col3], dtype=dtype).T


class TestReference(unittest.TestCase):
    def test_reference_wiki(self):
        """the reference implementation itself reproduces the wiki example"""
        arr = np.array(
            [[5, 4, 3], [2, 1, 4], [3, 4, 6], [4, 2, 8]], dtype=np.float64
        )
        expected = np.array(
            [
                [5.66666667, 5.16666667, 2.0],
                [2.0, 2.0, 3.0],
                [3.0, 5.16666667, 4.66666667],
                [4.66666667, 3.0, 5.66666667],
            ]
        )
        np.testing.assert_array_almost_equal(_reference_qnorm(arr), expected)


class TestEquivalence(unittest.TestCase):
    def test_random_sweep(self):
        """
        seeded sweep over shapes, dtypes, value distributions, NaNs, memory
        layouts, axes and ncpus
        """
        case = 0
        for shape in SHAPES:
            for dtype in DTYPES:
                for kind in KINDS:
                    case += 1
                    rng = np.random.default_rng(case)
                    nans = NANS[rng.integers(len(NANS))]
                    if not np.issubdtype(dtype, np.floating):
                        nans = 0.0
                    layout = LAYOUTS[rng.integers(len(LAYOUTS))]
                    axis = int(rng.integers(2))
                    ncpus = 3 if case % 5 == 0 else 1

                    msg = (
                        f"case={case} shape={shape} dtype="
                        f"{np.dtype(dtype).name} kind={kind} nans={nans} "
                        f"layout={layout} axis={axis} ncpus={ncpus}"
                    )
                    with self.subTest(msg):
                        arr = _make_array(rng, kind, shape, dtype, nans)
                        expected = _reference_qnorm(arr)
                        if axis == 1:
                            inp = _with_layout(arr, layout)
                        else:
                            inp = _with_layout(arr.T, layout)
                        result = qnorm.quantile_normalize(
                            inp, axis=axis, ncpus=ncpus
                        )
                        if axis == 0:
                            result = result.T
                        _assert_close(self, result, expected, msg)

    def test_specials(self):
        """infinities, signed zeros, NaNs, denormals and big values"""
        for dtype in [np.float32, np.float64]:
            arr = _special_array(dtype)
            expected = _reference_qnorm(arr)
            for layout in LAYOUTS:
                for ncpus in [1, 2]:
                    msg = f"dtype={np.dtype(dtype).name} layout={layout}"
                    with self.subTest(msg, ncpus=ncpus):
                        result = qnorm.quantile_normalize(
                            _with_layout(arr, layout), ncpus=ncpus
                        )
                        _assert_close(self, result, expected, msg)

    def test_float32_precision_loss(self):
        """int32 values above 2**24 are not exact in float32 and create ties"""
        rng = np.random.default_rng(0)
        arr = rng.integers(2**24, 2**30, size=(500, 3)).astype(np.int32)
        arr[:10] = 2**24 + 1  # all identical after rounding to float32
        _assert_close(
            self, qnorm.quantile_normalize(arr), _reference_qnorm(arr)
        )

    def test_empty(self):
        for shape in [(0, 3), (3, 0), (0, 0)]:
            for dtype in [np.float32, np.float64, np.int64]:
                with self.subTest(shape=shape, dtype=dtype):
                    result = qnorm.quantile_normalize(np.ones(shape, dtype))
                    self.assertEqual(result.shape, shape)
                    self.assertEqual(result.dtype, _qnorm_dtype(dtype))

    def test_target(self):
        rng = np.random.default_rng(1)
        for dtype in [np.float32, np.float64, np.int32]:
            arr = _make_array(rng, "mixed", (50, 4), dtype)
            targets = {
                "sorted": np.sort(rng.standard_normal(50)),
                "unsorted": rng.standard_normal(50),
                "list": list(rng.standard_normal(50)),
                "int": rng.integers(0, 10, 50),
                "float32": rng.standard_normal(50).astype(np.float32),
            }
            for name, target in targets.items():
                for ncpus in [1, 2]:
                    with self.subTest(dtype=dtype, target=name, ncpus=ncpus):
                        result = qnorm.quantile_normalize(
                            arr, target=target, ncpus=ncpus
                        )
                        expected = _reference_qnorm(arr, target=target)
                        _assert_close(self, result, expected)

    def test_target_axis0(self):
        rng = np.random.default_rng(2)
        arr = _make_array(rng, "mixed", (30, 5), np.float64)
        target = rng.standard_normal(30)
        np.testing.assert_allclose(
            qnorm.quantile_normalize(arr.T, axis=0, target=target).T,
            _reference_qnorm(arr, target=target),
        )


class TestExtremeValues(unittest.TestCase):
    """
    The rank means are summed in float64, in a fixed order that differs from
    numpy's mean (and the summation of earlier versions of qnorm). So instead
    of comparing against another summation, compare against the exactly
    rounded means, with the error that any summation order can make.
    """

    def test_extreme_values(self):
        rng = np.random.default_rng(16)
        for dtype in [np.float32, np.float64]:
            for kind in ["huge", "tiny", "range", "cancel"]:
                # 21 columns: two full groups of _TARGET_COLS and a partial one
                arr = _extreme_array(rng, kind, (500, 21), dtype)
                expected, tol = _exact_qnorm(arr)
                single = qnorm.quantile_normalize(arr, ncpus=1)
                for ncpus in [1, 3]:
                    msg = f"dtype={np.dtype(dtype).name} kind={kind}"
                    with self.subTest(msg, ncpus=ncpus):
                        result = qnorm.quantile_normalize(arr, ncpus=ncpus)
                        self.assertEqual(result.dtype, dtype)
                        error = np.abs(result.astype(np.float64) - expected)
                        np.testing.assert_array_less(error, tol)
                        np.testing.assert_array_equal(result, single)

    def test_exact_qnorm(self):
        """the exact reference agrees with the naive reference"""
        rng = np.random.default_rng(17)
        arr = _make_array(rng, "mixed", (200, 9), np.float64)
        expected, tol = _exact_qnorm(arr)
        np.testing.assert_allclose(expected, _reference_qnorm(arr))
        self.assertTrue(np.all(tol < 1e-9))


class TestInvariants(unittest.TestCase):
    def test_input_not_mutated(self):
        rng = np.random.default_rng(3)
        for dtype in [np.float32, np.float64, np.int32]:
            for layout in LAYOUTS:
                for axis in [0, 1]:
                    arr = _with_layout(
                        _make_array(rng, "mixed", (200, 6), dtype, 0.0),
                        layout,
                    )
                    if np.issubdtype(dtype, np.floating):
                        arr[3, 2] = np.nan
                    before = arr.copy()
                    qnorm.quantile_normalize(arr, axis=axis, ncpus=2)
                    with self.subTest(dtype=dtype, layout=layout, axis=axis):
                        np.testing.assert_array_equal(arr, before)

    def test_dataframe_not_mutated(self):
        df = pd.DataFrame(
            np.random.default_rng(4).standard_normal((20, 3)),
            columns=["a", "b", "c"],
        )
        before = df.copy()
        qnorm.quantile_normalize(df, ncpus=2)
        pd.testing.assert_frame_equal(df, before)

    def test_output_dtype(self):
        expected = {
            np.int8: np.float64,
            np.int16: np.float64,
            np.int32: np.float32,
            np.int64: np.float64,
            np.uint8: np.float64,
            np.uint32: np.float64,
            np.float16: np.float64,
            np.float32: np.float32,
            np.float64: np.float64,
        }
        for dtype, out_dtype in expected.items():
            with self.subTest(dtype=dtype):
                arr = np.arange(12).reshape(4, 3).astype(dtype)
                self.assertEqual(qnorm.quantile_normalize(arr).dtype, out_dtype)
                self.assertEqual(
                    qnorm.quantile_normalize(arr, target=np.arange(4)).dtype,
                    out_dtype,
                )

    def test_output_layout(self):
        """the memory layout of the output follows the input"""
        arr = np.random.default_rng(5).standard_normal((30, 4))
        for axis in [0, 1]:
            out = qnorm.quantile_normalize(np.ascontiguousarray(arr), axis)
            self.assertTrue(out.flags.c_contiguous)
            out = qnorm.quantile_normalize(np.asfortranarray(arr), axis)
            self.assertTrue(out.flags.f_contiguous)

    def test_output_is_new_array(self):
        for dtype in [np.float32, np.float64]:
            arr = np.random.default_rng(6).standard_normal((30, 4))
            arr = arr.astype(dtype)
            for axis in [0, 1]:
                out = qnorm.quantile_normalize(arr, axis=axis)
                self.assertFalse(np.shares_memory(out, arr))

    def test_ncpus_same_result(self):
        """the number of cpus does not change the result"""
        rng = np.random.default_rng(7)
        for kind in KINDS:
            for dtype in [np.float32, np.float64]:
                arr = _make_array(rng, kind, (3000, 13), dtype, 0.05)
                reference = qnorm.quantile_normalize(arr, ncpus=1)
                for ncpus in [2, 3, 8]:
                    with self.subTest(kind=kind, dtype=dtype, ncpus=ncpus):
                        _assert_close(
                            self,
                            qnorm.quantile_normalize(arr, ncpus=ncpus),
                            reference,
                        )

    def test_same_distribution(self):
        """without ties every column ends up with the same sorted values"""
        arr = np.random.default_rng(8).standard_normal((500, 7))
        result = qnorm.quantile_normalize(arr)
        sorted_result = np.sort(result, axis=0)
        for col in range(1, arr.shape[1]):
            np.testing.assert_allclose(
                sorted_result[:, col], sorted_result[:, 0]
            )
        np.testing.assert_allclose(
            sorted_result[:, 0], np.sort(arr, axis=0).mean(axis=1)
        )

    def test_idempotent(self):
        """without ties normalizing twice changes nothing"""
        arr = np.random.default_rng(9).standard_normal((500, 7))
        once = qnorm.quantile_normalize(arr)
        np.testing.assert_allclose(qnorm.quantile_normalize(once), once)

    def test_rank_preserved(self):
        """the order within a column never changes"""
        rng = np.random.default_rng(10)
        arr = _make_array(rng, "mixed", (1000, 8), np.float64)
        result = qnorm.quantile_normalize(arr)
        for col in range(arr.shape[1]):
            order = np.argsort(arr[:, col], kind="stable")
            self.assertTrue(np.all(np.diff(result[order, col]) >= 0))

    def test_pandas_matches_numpy(self):
        rng = np.random.default_rng(11)
        arr = _make_array(rng, "mixed", (100, 5), np.float64, 0.05)
        df = pd.DataFrame(
            arr,
            index=[f"row{i}" for i in range(100)],
            columns=[f"col{i}" for i in range(5)],
        )
        for axis in [0, 1]:
            for ncpus in [1, 2]:
                with self.subTest(axis=axis, ncpus=ncpus):
                    result = qnorm.quantile_normalize(
                        df, axis=axis, ncpus=ncpus
                    )
                    self.assertIsInstance(result, pd.DataFrame)
                    pd.testing.assert_index_equal(result.index, df.index)
                    pd.testing.assert_index_equal(result.columns, df.columns)
                    np.testing.assert_allclose(
                        result.values,
                        qnorm.quantile_normalize(arr, axis=axis),
                    )


class TestErrors(unittest.TestCase):
    def setUp(self):
        self.arr = np.random.default_rng(12).standard_normal((4, 2))

    def test_non_numeric_data(self):
        for bad in [np.array([["a", "b"], ["c", "d"]]), np.ones((3, 2), bool)]:
            with self.subTest(dtype=bad.dtype):
                self.assertRaises(ValueError, qnorm.quantile_normalize, bad)

    def test_unsupported_type(self):
        self.assertRaises(
            NotImplementedError, qnorm.quantile_normalize, [[1, 2], [3, 4]]
        )

    def test_bad_axis(self):
        for axis in [2, -1]:
            self.assertRaises(
                ValueError, qnorm.quantile_normalize, self.arr, axis=axis
            )

    def test_bad_target(self):
        bad_targets = [
            np.arange(3),
            np.ones((4, 1)),
            np.array(list("abcd")),
        ]
        for target in bad_targets:
            with self.subTest(target=target):
                self.assertRaises(
                    ValueError,
                    qnorm.quantile_normalize,
                    self.arr,
                    target=target,
                )

    def test_bad_ncpus(self):
        for ncpus in [0, -1, 1.5, 2.0, "2", None, True, np.float64(2)]:
            with self.subTest(ncpus=ncpus):
                self.assertRaises(
                    ValueError, qnorm.quantile_normalize, self.arr, ncpus=ncpus
                )
                # checked before the (here non-existing) file is read
                self.assertRaises(
                    ValueError,
                    qnorm.incremental_quantile_normalize,
                    "missing.csv",
                    "missing_out.csv",
                    ncpus=ncpus,
                )

    def test_numpy_integer_ncpus(self):
        expected = qnorm.quantile_normalize(self.arr)
        for ncpus in [np.int64(2), np.int32(1), np.uint8(3)]:
            with self.subTest(ncpus=ncpus):
                np.testing.assert_array_equal(
                    qnorm.quantile_normalize(self.arr, ncpus=ncpus), expected
                )


class TestDeterminism(unittest.TestCase):
    def test_ncpus_bit_identical(self):
        """the number of cpus does not change the result, not even a bit"""
        rng = np.random.default_rng(14)
        for shape in [(3000, 13), (500, 40)]:
            for kind in ["unique", "mixed"]:
                for dtype in [np.float32, np.float64]:
                    arr = _make_array(rng, kind, shape, dtype, 0.05)
                    reference = qnorm.quantile_normalize(arr, ncpus=1)
                    for ncpus in [2, 3, 8]:
                        with self.subTest(
                            shape=shape, kind=kind, dtype=dtype, ncpus=ncpus
                        ):
                            np.testing.assert_array_equal(
                                qnorm.quantile_normalize(arr, ncpus=ncpus),
                                reference,
                            )

    def test_small_blocks(self):
        """copy between memory layouts in many small blocks"""
        rng = np.random.default_rng(15)
        for block_bytes in [40, 100]:
            with mock.patch.object(QN_MODULE, "_BLOCK_BYTES", block_bytes):
                for layout in LAYOUTS:
                    for axis in [0, 1]:
                        for ncpus in [1, 3]:
                            arr = _make_array(
                                rng, "mixed", (101, 6), np.float64, 0.05
                            )
                            expected = _reference_qnorm(arr)
                            inp = arr if axis == 1 else arr.T
                            result = qnorm.quantile_normalize(
                                _with_layout(inp, layout), axis, ncpus=ncpus
                            )
                            if axis == 0:
                                result = result.T
                            with self.subTest(
                                block_bytes=block_bytes,
                                layout=layout,
                                axis=axis,
                                ncpus=ncpus,
                            ):
                                _assert_close(self, result, expected)


class TestPandasDtypes(unittest.TestCase):
    """
    DataFrames of every numeric dtype. Earlier versions of qnorm wrote the
    results back into a copy of the input, which pandas >= 3.0 refuses for
    every dtype other than float64.
    """

    def test_dtypes(self):
        values = np.random.default_rng(16).integers(0, 50, (40, 3))
        index = [f"row{i}" for i in range(40)]
        expected_dtypes = {
            "int64": np.float64,
            "int32": np.float64,
            "uint8": np.float64,
            "object": np.float64,
            "float32": np.float32,
            "float64": np.float64,
        }
        for dtype, out_dtype in expected_dtypes.items():
            with self.subTest(dtype=dtype):
                df = pd.DataFrame(
                    values.astype(dtype), index=index, columns=list("abc")
                )
                result = qnorm.quantile_normalize(df)
                self.assertTrue((result.dtypes == out_dtype).all())
                pd.testing.assert_index_equal(result.index, df.index)
                pd.testing.assert_index_equal(result.columns, df.columns)
                np.testing.assert_allclose(
                    result.values,
                    qnorm.quantile_normalize(values.astype(out_dtype)),
                )

    def test_mixed_dtypes(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [1.5, 0.5, 2.5, 2.0],
                "c": np.array([3, 1, 2, 4], dtype=np.float32),
            }
        )
        result = qnorm.quantile_normalize(df)
        self.assertTrue((result.dtypes == np.float64).all())
        np.testing.assert_allclose(
            result.values,
            qnorm.quantile_normalize(df.values.astype(np.float64)),
        )


class TestGolden(unittest.TestCase):
    def test_golden(self):
        """reproduce the outputs of an earlier implementation of qnorm"""
        golden = np.load(GOLDEN)
        n_cases = int(golden["n_cases"])
        self.assertGreater(n_cases, 0)
        for case in range(n_cases):
            arr = golden[f"input_{case}"]
            axis = int(golden[f"axis_{case}"])
            if f"target_{case}" in golden:
                target = golden[f"target_{case}"]
            else:
                target = None
            expected = golden[f"output_{case}"]
            for ncpus in [1, 2]:
                with self.subTest(case=case, ncpus=ncpus):
                    result = qnorm.quantile_normalize(
                        arr, axis=axis, target=target, ncpus=ncpus
                    )
                    _assert_close(self, result, expected, f"case {case}")


class TestIncremental(unittest.TestCase):
    """the file-based implementation equals the in-memory implementation"""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(13)
        n_rows, n_cols = 300, 7
        values = np.hstack(
            [
                rng.standard_normal((n_rows, 3)),
                rng.integers(0, 5, (n_rows, 2)),
                rng.negative_binomial(2, 0.05, (n_rows, 2)),
            ]
        )
        cls.df = pd.DataFrame(
            values,
            index=[f"r{i}" for i in range(n_rows)],
            columns=[f"c{i}" for i in range(n_cols)],
        )
        cls.expected = qnorm.quantile_normalize(cls.df).values
        cls.df.to_csv("test_equiv.csv")
        cls.df.to_hdf(
            "test_equiv.hdf",
            key="qnorm",
            format="table",
            data_columns=True,
            mode="w",
        )
        cls.df.to_parquet("test_equiv.parquet")

    @classmethod
    def tearDownClass(cls):
        for ext in ["csv", "hdf", "parquet"]:
            for name in [f"test_equiv.{ext}", f"test_equiv_out.{ext}"]:
                if os.path.exists(name):
                    os.remove(name)

    def _check(self, ext, read, rtol):
        for rowchunksize, colchunksize in [(1000, 1000), (7, 3), (64, 2)]:
            for ncpus in [1, 2]:
                with self.subTest(
                    rowchunksize=rowchunksize,
                    colchunksize=colchunksize,
                    ncpus=ncpus,
                ):
                    qnorm.incremental_quantile_normalize(
                        f"test_equiv.{ext}",
                        f"test_equiv_out.{ext}",
                        rowchunksize=rowchunksize,
                        colchunksize=colchunksize,
                        ncpus=ncpus,
                    )
                    result = read(f"test_equiv_out.{ext}")
                    self.assertEqual(list(result.columns), list(self.df))
                    self.assertEqual(
                        [str(i) for i in result.index],
                        [str(i) for i in self.df.index],
                    )
                    np.testing.assert_allclose(
                        result.values, self.expected, rtol=rtol, atol=1e-4
                    )

    def test_csv(self):
        self._check("csv", lambda f: pd.read_csv(f, index_col=0), 1e-5)

    def test_hdf(self):
        self._check("hdf", pd.read_hdf, 1e-6)

    def test_parquet(self):
        self._check("parquet", pd.read_parquet, 1e-12)


if __name__ == "__main__":
    unittest.main()
