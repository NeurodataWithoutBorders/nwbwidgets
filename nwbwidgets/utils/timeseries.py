from pynwb import TimeSeries

import numpy as np
from bisect import bisect


def get_timeseries_tt(node: TimeSeries, istart=0, istop=None):
    if node.timestamps is not None:
        return node.timestamps[istart:istop]
    else:
        if istop is None:
            return np.arange(istart, len(node.data) - 1) / node.rate + node.starting_time
        elif istop > 0:
            return np.arange(istart, istop) / node.rate + node.starting_time
        else:
            return np.arange(istart, len(node.data) + istop - 1) / node.rate + node.starting_time


def get_timeseries_maxt(node: TimeSeries):
    if node.timestamps is not None:
        return node.timestamps[-1]
    else:
        return len(node.data) / node.rate + node.starting_time


def get_timeseries_mint(node: TimeSeries):
    if node.timestamps is not None:
        return node.timestamps[0]
    else:
        return node.starting_time


def get_timeseries_in_units(node: TimeSeries, istart=0, istop=-1):
    data = node.data[istart:istop]
    if node.conversion and np.isfinite(node.conversion):
        data = data * node.conversion
        unit = node.unit
    else:
        unit = None
    return data, unit


def timeseries_time_to_ind(node: TimeSeries, time, ind_min=None, ind_max=None):
    if node.timestamps is not None:
        kwargs = dict()
        if ind_min is not None:
            kwargs.update(lo=ind_min)
        if ind_max is not None:
            kwargs.update(hi=ind_max)
        return bisect(node.timestamps, time, **kwargs)
    else:
        return int(np.ceil((time - node.starting_time) * node.rate))
