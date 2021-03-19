"""Top-level package for qnorm."""
from .quantile_normalize import quantile_normalize

__all__ = ["quantile_normalize"]

try:
    from .quantile_normalize import incremental_quantile_normalize  # noqa: F401

    __all__.append("incremental_quantile_normalize")
except ImportError:
    pass

__author__ = "Maarten van der Sande"
__email__ = "maartenvandersande[at]gmail.com"
__version__ = "0.7.0"
