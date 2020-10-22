"""Top-level package for qnorm."""
from .quantile_normalize import quantile_normalize
__all__ = ["quantile_normalize"]

try:
    from .quantile_normalize import quantile_normalize_file
    __all__.append("quantile_normalize_file")
except:
    pass

__author__ = "Maarten van der Sande"
__email__ = "maartenvandersande[at]gmail.com"
__version__ = "0.6.1"
