"""Console script for qnorm."""
import argparse
import sys

import pandas as pd

import qnorm


def main():
    """Console script for qnorm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("table", metavar="FILE")
    args = parser.parse_args()

    print(args)


if __name__ == "__main__":
    sys.exit(main())
