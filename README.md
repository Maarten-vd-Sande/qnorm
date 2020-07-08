# qnorm
[![PyPI version](https://badge.fury.io/py/qnorm.svg)](https://badge.fury.io/py/qnorm)
[![Anaconda version](https://anaconda.org/conda-forge/qnorm/badges/version.svg)](https://anaconda.org/conda-forge/qnorm/badges/version.svg)

quantile normalization made easy.

## Quick example

We recreate the example of [Wikipedia](https://en.wikipedia.org/wiki/Quantile_normalization):

```
import numpy as np
import qnorm

vals = np.array([
    [5, 4, 3],
    [2, 1, 4],
    [3, 4, 6],
    [4, 2, 8]])

print(qnorm.quantile_normalize(vals))
```

which prints this:

```
>>> [[5.66666667 5.16666667 2.        ]
     [2.         2.         3.        ]
     [3.         5.16666667 4.66666667]
     [4.66666667 3.         5.66666667]]
```


## Installation

### pip

```
pip install qnorm
```

### conda

Installing qnorm from the conda-forge channel can be achieved by adding conda-forge to your channels with:

```
conda config --add channels conda-forge
```

Once the conda-forge channel has been enabled, qnorm can be installed with:

```
conda install qnorm
```
