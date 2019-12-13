from pynwb import TimeSeries

import numpy as np
from bisect import bisect


def get_timeseries_tt(node: TimeSeries, istart=0, istop=None):
    if node.timestamps is not None:
        return node.timestamps[istart:istop]
    else:
        if istop is None:
            return np.arange(istart, len(node.data)) / node.rate + node.starting_time
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


def get_timeseries_in_units(node: TimeSeries, istart=None, istop=None):
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


def align_by_times(timeseries: TimeSeries, starts, stops):
    """
    Args:
        timeseries: TimeSeries
        starts: array-like
        stops: array-like
    Returns:
        np.array(shape=(n_trials, n_time, ...))
    """
    out = []
    for istart, istop in zip(starts, stops):
        ind_start = timeseries_time_to_ind(timeseries, istart)
        ind_stop = timeseries_time_to_ind(timeseries, istop, ind_min=ind_start)
        out.append(timeseries.data[ind_start:ind_stop])
    return np.array(out)


def align_by_trials(timeseries: TimeSeries, start_label='start_time',
                    stop_label=None, before=0., after=1.):
    """
    Args:
        timeseries: TimeSeries
        start_label: str
            default: 'start_time'
        stop_label: str
            default: None (just align to start_time)
        before: float
            time after start_label in secs (positive goes back in time)
        after: float
            time after stop_label in secs (positive goes forward in time)
    Returns:
        np.array(shape=(n_trials, n_time, ...))
    """
    trials = timeseries.get_ancestor('NWBFile').trials
    return align_by_time_intervals(timeseries, trials, start_label, stop_label, before, after)


def align_by_time_intervals(timeseries: TimeSeries, intervals, start_label='start_time',
                            stop_label='stop_time', before=0., after=0.):
    """
    Args:
        timeseries: pynwb.TimeSeries
        intervals: pynwb.epoch.TimeIntervals
        start_label: str
            default: 'start_time'
        stop_label: str
            default: 'stop_time'
        before: float
            time after start_label in secs (positive goes back in time)
        after: float
            time after stop_label in secs (positive goes forward in time)
    Returns:
        np.array(shape=(n_trials, n_time, ...))
    """
    if stop_label is None:
        stop_label = 'start_time'

    starts = np.array(intervals[start_label][:]) - before
    stops = np.array(intervals[stop_label][:]) + after
    return align_by_times(timeseries, starts, stops)
