from bisect import bisect, bisect_left

import numpy as np

from pynwb import TimeSeries


def get_timeseries_tt(node: TimeSeries, istart=0, istop=None) -> np.ndarray:
    """
    For any TimeSeries, return timestamps. If the TimeSeries uses starting_time and rate, the timestamps will be
    generated.

    Parameters
    ----------
    node: pynwb.TimeSeries
    istart: int, optional
        Optionally sub-select the returned times - lower bound
    istop: int, optional
        Optionally sub-select the returned times - upper bound

    Returns
    -------
    numpy.ndarray

    """
    if node.timestamps is not None:
        return node.timestamps[istart:istop]
    else:
        if not np.isfinite(node.starting_time):
            starting_time = 0
        else:
            starting_time = node.starting_time
        if istop is None:
            return np.arange(istart, len(node.data)) / node.rate + starting_time
        elif istop > 0:
            return np.arange(istart, istop) / node.rate + starting_time
        else:
            return (
                np.arange(istart, len(node.data) + istop - 1) / node.rate
                + starting_time
            )


def get_timeseries_maxt(node: TimeSeries) -> float:
    """
    Returns the maximum time of any TimeSeries

    Parameters
    ----------
    node: pynwb.TimeSeries

    Returns
    -------
    float

    """
    if node.timestamps is not None:
        return node.timestamps[-1]
    elif np.isnan(node.starting_time):
        return (len(node.data) - 1) / node.rate
    else:
        return (len(node.data) - 1) / node.rate + node.starting_time


def get_timeseries_mint(node: TimeSeries) -> float:
    """
    Returns the minimum time of any TimeSeries

    Parameters
    ----------
    node: pynwb.TimeSeries

    Returns
    -------
    float

    """
    if node.timestamps is not None:
        return node.timestamps[0]
    elif np.isnan(node.starting_time):
        return 0
    else:
        return node.starting_time


def get_timeseries_in_units(node: TimeSeries, istart=None, istop=None):
    """
    Convert data into the designated units

    Parameters
    ----------
    node: pynwb.TimeSeries
    istart: int
    istop: int

    Returns
    -------
    numpy.ndarray, str

    """
    data = node.data[istart:istop]
    if node.conversion and np.isfinite(node.conversion):
        data = data * node.conversion
        unit = node.unit
    else:
        unit = None
    return data, unit


def timeseries_time_to_ind(node: TimeSeries, time, ind_min=None, ind_max=None) -> int:
    """
    Get the index of a certain time for any TimeSeries. For TimeSeries that use timestamps, bisect is used. You can
    optionally provide ind_min and ind_max to constrain the search.

    Parameters
    ----------
    node: pynwb.TimeSeries
    time: float
    ind_min: int, optional
    ind_max: int, optional

    Returns
    -------

    """
    if node.timestamps is not None:
        kwargs = dict()
        if ind_min is not None:
            kwargs.update(lo=ind_min)
        if ind_max is not None:
            kwargs.update(hi=ind_max)
        id_found = bisect_left(node.timestamps, time, **kwargs)
        return id_found if id_found < len(node.data) else len(node.data) - 1
    else:
        if np.isnan(node.starting_time):
            starting_time = 0
        else:
            starting_time = node.starting_time
        id_found = int(np.ceil((time - starting_time) * node.rate))
        return id_found if id_found < len(node.data) else len(node.data) - 1


def bisect_timeseries_by_times(
    timeseries: TimeSeries, starts, duration: float, traces=None
):
    """
    Parameters
    ----------
    timeseries: TimeSeries
    starts: iterable
        time at which to bisect
    duration: float
        duration of window after start
    traces: int
        index into the second dim of data
    Returns
    -------
    out: list
        list with bisected arrays from data
    """
    out = []
    for start in starts:
        if timeseries.rate is not None:
            idx_start = int((start - timeseries.starting_time) * timeseries.rate)
            idx_stop = int(idx_start + duration * timeseries.rate)
        else:
            idx_start = bisect(timeseries.timestamps, start)
            idx_stop = bisect(timeseries.timestamps, start + duration, lo=idx_start)
        if len(timeseries.data.shape) > 1 and traces is not None:
            out.append(timeseries.data[idx_start:idx_stop, traces])
        else:
            out.append(timeseries.data[idx_start:idx_stop])
    return out


def align_by_times_with_timestamps(
    timeseries: TimeSeries, starts, duration: float, traces=None
):
    """
    Parameters
    ----------
    timeseries: TimeSeries
        timeseries with variable timestamps
    starts: array-like
        starts in seconds
    duration: float
        duration in seconds
    Returns
    -------
    out: list
        list: length=(n_trials); list[0]: array, shape=(n_time, ...)
    """
    assert timeseries.timestamps is not None, "supply timeseries with timestamps"
    return bisect_timeseries_by_times(timeseries, starts, duration, traces)


def align_by_times_with_rate(
    timeseries: TimeSeries, starts, duration: float, traces=None
):
    """
    Parameters
    ----------
    timeseries: TimeSeries
        timeseries with variable timestamps
    starts: array-like
        starts in seconds
    duration: float
        duration in seconds
    Returns
    -------
    out: list
        list: length=(n_trials); list[0]: array, shape=(n_time, ...)
    """
    assert timeseries.rate is not None, "supply timeseries with start_time and rate"
    return np.array(bisect_timeseries_by_times(timeseries, starts, duration, traces))


def align_timestamps_by_trials(
    timeseries: TimeSeries, starts, before: float, after: float
):
    """
    Parameters
    ----------
    timeseries: TimeSeries
        timeseries with variable timestamps
    starts: array-like
        starts in seconds
    duration: float
        duration in seconds
    Returns
    -------
    out: list
        list: length=(n_trials); list[0]: array, shape=(n_time, ...)
    """
    assert timeseries.timestamps is not None, "supply timeseries with timestamps"
    out = []
    for start in starts:
        idx_start = bisect(timeseries.timestamps, start)
        idx_stop = bisect(timeseries.timestamps, start + before + after, lo=idx_start)
        out.append(timeseries.timestamps[idx_start:idx_stop])
    return [list(np.array(i) - i[0] - before) for i in out]


def align_by_trials(
    timeseries: TimeSeries,
    start_label="start_time",
    before=0.0,
    after=1.0,
):
    """
    Args:
        timeseries: TimeSeries
        start_label: str
            default: 'start_time'
        before: float
            time after start_label in secs (positive goes back in time)
        after: float
            time after stop_label in secs (positive goes forward in time)
    Returns:
        np.array(shape=(n_trials, n_time, ...))
    """
    trials = timeseries.get_ancestor("NWBFile").trials
    return align_by_time_intervals(timeseries, trials, start_label, before, after)


def align_by_time_intervals(
    timeseries: TimeSeries,
    intervals,
    start_label="start_time",
    before=0.0,
    after=0.0,
    traces=None,
):
    """
    Args:
        timeseries: pynwb.TimeSeries
        intervals: pynwb.epoch.TimeIntervals
        start_label: str
            default: 'start_time'
        before: float
            time after start_label in secs (positive goes back in time)
        after: float
            time after stop_label in secs (positive goes forward in time)
        timestamps: bool
            if alignment uses timestamps or constant rate and starting time in TimeSeries
    Returns:
        np.array(shape=(n_trials, n_time, ...))
    """

    starts = np.array(intervals[start_label][:]) - before
    if timeseries.rate is not None:
        return align_by_times_with_rate(
            timeseries, starts, duration=after + before, traces=traces
        )
    else:
        return align_by_times_with_timestamps(
            timeseries, starts, duration=after + before, traces=traces
        )
