# qnorm
[![PyPI version](https://badge.fury.io/py/qnorm.svg)](https://badge.fury.io/py/qnorm)
[![Anaconda version](https://anaconda.org/conda-forge/qnorm/badges/version.svg)](https://anaconda.org/conda-forge/qnorm/badges/version.svg)
![tests](https://github.com/Maarten-vd-Sande/qnorm/workflows/tests/badge.svg)

quantile normalization made easy.

## Quick example

We recreate the example of [Wikipedia](https://en.wikipedia.org/wiki/Quantile_normalization):

```
import pandas as pd
import qnorm

df = pd.DataFrame({'C1': {'A': 5, 'B': 2, 'C': 3, 'D': 4},
                   'C2': {'A': 4, 'B': 1, 'C': 4, 'D': 2},
                   'C3': {'A': 3, 'B': 4, 'C': 6, 'D': 8}})

print(qnorm.quantile_normalize(df))
```

which is what we expect:

```
         C1        C2        C3
A  5.666667  5.166667  2.000000
B  2.000000  2.000000  3.000000
C  3.000000  5.166667  4.666667
D  4.666667  3.000000  5.666667
```

The function quantile_normalize also accepts numpy arrays. 

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

### local

clone the repository

```
git clone https://github.com/Maarten-vd-Sande/qnorm
```

And install it

```
cd qnorm
pip install .
```
