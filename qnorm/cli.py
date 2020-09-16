"""Console script for qnorm."""
import argparse
import sys
import warnings

try:
    import pandas as pd
except ModuleNotFoundError:
    raise ImportError(
        "To make use of the CLI of qnorm pandas needs to be installed!"
    )

import qnorm


def main():
    """Console script for qnorm."""
    parser = argparse.ArgumentParser(
        description="Quantile normalization from the CLI!"
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"qnorm: v{qnorm.__version__}",
    )
    parser.add_argument(
        "table", help="input csv/tsv file which will be quantile normalized"
    )
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inferred_sep = pd.read_csv(
            args.table, sep=None, iterator=True
        )._engine.data.dialect.delimiter

    df = pd.read_csv(args.table, index_col=0, sep=inferred_sep, comment="#")
    qnorm_df = qnorm.quantile_normalize(df)

    print(qnorm_df.to_csv(sep=inferred_sep))


if __name__ == "__main__":
    sys.exit(main())
