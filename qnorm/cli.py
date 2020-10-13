"""Console script for qnorm."""
import argparse
import sys

from .util import get_delim

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

    delimiter = get_delim(args.table)

    df = pd.read_csv(args.table, index_col=0, sep=delimiter, comment="#")
    qnorm_df = qnorm.quantile_normalize(df)

    print(qnorm_df.to_csv(sep=delimiter))


if __name__ == "__main__":
    sys.exit(main())
