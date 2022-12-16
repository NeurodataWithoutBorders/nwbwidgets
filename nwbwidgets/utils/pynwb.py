import numpy as np
from pynwb import NWBContainer


def robust_unique(a):
    if isinstance(a[0], NWBContainer):
        return np.unique([x.name for x in a])
    return np.unique(a)
