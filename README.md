# qnorm
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

```
conda install -c conda-forge qnorm
```
