import numpy as np
import pynwb


def robust_unique(a):
    if isinstance(a[0], pynwb.NWBContainer):
        return np.unique([x.name for x in a])
    return np.unique(a)