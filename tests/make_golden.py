"""
Generate tests/data/golden.npz: inputs and outputs of the current qnorm
implementation, used by tests/test_equivalence.py:TestGolden as a regression
lock.

Only regenerate this file when the behaviour of qnorm is deliberately
changed:

    python -m tests.make_golden
"""

import os

import numpy as np

import qnorm
from tests.test_equivalence import GOLDEN, _make_array, _special_array


def _cases():
    rng = np.random.default_rng(2024)
    wiki = np.array(
        [[5, 4, 3], [2, 1, 4], [3, 4, 6], [4, 2, 8]], dtype=np.float64
    )
    ties = np.array(
        [[2, 2], [2, 2], [2, 1], [2, 3.5], [6, 5], [1, 1]], dtype=np.float64
    )
    signed_zeros = rng.integers(-2, 3, (40, 4)).astype(np.float64)
    zeros = signed_zeros == 0
    signed_zeros[zeros] = np.where(rng.random(zeros.sum()) < 0.5, 0.0, -0.0)
    big_int32 = rng.integers(2**24, 2**30, (300, 3)).astype(np.int32)

    yield dict(data=wiki)
    yield dict(data=ties)
    yield dict(data=_make_array(rng, "ties", (500, 6), np.int32))
    yield dict(data=rng.integers(0, 100, (500, 6)).astype(np.int32))
    yield dict(data=_make_array(rng, "unique", (300, 4), np.float32))
    yield dict(data=_make_array(rng, "unique", (300, 4), np.float64))
    yield dict(data=_make_array(rng, "counts", (400, 5), np.int64))
    yield dict(data=_make_array(rng, "mixed", (400, 8), np.float64, 0.1))
    yield dict(data=_make_array(rng, "mixed", (50, 3), np.float32, "allnan"))
    yield dict(data=_special_array(np.float64))
    yield dict(data=_special_array(np.float32))
    yield dict(data=big_int32)
    yield dict(data=_make_array(rng, "unique", (300, 4), np.uint8))
    yield dict(data=_make_array(rng, "unique", (5, 300), np.float64), axis=0)
    yield dict(data=_make_array(rng, "ties", (6, 200), np.int32), axis=0)
    yield dict(
        data=_make_array(rng, "mixed", (100, 4), np.float64),
        target=rng.standard_normal(100),
    )
    yield dict(
        data=_make_array(rng, "mixed", (100, 4), np.float32, 0.05),
        target=rng.standard_normal(100),
    )
    yield dict(data=np.array([[3.0]]))
    yield dict(data=np.array([[3.0, 1.0, 2.0, 2.0, 5.0]]))
    yield dict(data=np.array([[3.0], [1.0], [2.0], [2.0], [5.0]]))
    yield dict(data=signed_zeros)
    yield dict(data=_make_array(rng, "constant", (30, 3), np.float64))


def main():
    arrays = {}
    n_cases = 0
    for case, kwargs in enumerate(_cases()):
        axis = kwargs.get("axis", 1)
        target = kwargs.get("target")
        arrays[f"input_{case}"] = kwargs["data"]
        arrays[f"axis_{case}"] = np.array(axis)
        if target is not None:
            arrays[f"target_{case}"] = target
        arrays[f"output_{case}"] = qnorm.quantile_normalize(
            kwargs["data"], axis=axis, target=target
        )
        n_cases += 1
    arrays["n_cases"] = np.array(n_cases)

    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    np.savez_compressed(GOLDEN, **arrays)
    print(f"wrote {n_cases} cases to {GOLDEN}")


if __name__ == "__main__":
    main()
