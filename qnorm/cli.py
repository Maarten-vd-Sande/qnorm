"""Console script for qnorm."""
import argparse
import sys
import warnings

import pandas as pd

import qnorm


def main():
    """Console script for qnorm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("table")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inferred_sep = pd.read_csv(
            args.table, sep=None, iterator=True
        )._engine.data.dialect.delimiter

    df = pd.read_csv(args.table, index_col=0, sep=inferred_sep)
    qnorm_df = qnorm.quantile_normalize(df)

    print(qnorm_df.to_csv(sep=inferred_sep))


if __name__ == "__main__":
    sys.exit(main())
