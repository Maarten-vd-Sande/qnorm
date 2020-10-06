import numpy as np

# functions needed for parallel sorting
var_dict = {}


def worker_init(X, X_dtype, X_shape):
    """
    helper function to pass our reference of X to the sorter
    """
    var_dict["X"] = X
    var_dict["X_dtype"] = X_dtype
    var_dict["X_shape"] = X_shape


def worker_sort(i):
    """
    argsort a single axis
    """
    X_np = np.frombuffer(var_dict["X"], dtype=var_dict["X_dtype"]).reshape(
        var_dict["X_shape"]
    )
    return np.argsort(X_np[:, i])
