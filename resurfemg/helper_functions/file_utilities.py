import numpy as np
import pandas as pd

_DF_TAG = "__dataframe__"


def _json_default(obj):
    """Serialize types json can't handle natively."""
    if isinstance(obj, pd.DataFrame):
        # Tagged so the decoder can rebuild it; "split" keeps column/index order
        return {_DF_TAG: obj.to_dict(orient="split")}
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # one C-level call, works for 1D and 2D
    if isinstance(obj, np.generic):
        return obj.item()  # np.float64, np.int64, np.bool_ -> Python scalars
    msg = f"Object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)


def _json_object_hook(d):
    """Rebuild tagged DataFrames during json.load."""
    if _DF_TAG in d:
        return pd.DataFrame(**d[_DF_TAG])
    return d
