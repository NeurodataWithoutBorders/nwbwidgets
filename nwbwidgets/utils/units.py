import numpy as np
import pynwb

from bisect import bisect_right, bisect_left


def get_spike_times(units: pynwb.misc.Units, index, in_interval):
    """Use bisect methods to efficiently retrieve spikes from a given unit in a given interval

    Parameters
    ----------
    units: pynwb.misc.Units
    index: int
    in_interval: start and stop times

    Returns
    -------

    """
    st = units['spike_times']
    unit_start = 0 if index == 0 else st.data[index - 1]
    unit_stop = st.data[index]
    start_time, stop_time = in_interval

    ind_start = bisect_left(st.target, start_time, unit_start, unit_stop)
    ind_stop = bisect_right(st.target, stop_time, ind_start, unit_stop)

    return np.asarray(st.target[ind_start:ind_stop])


def get_min_spike_time(units: pynwb.misc.Units):
    """Efficiently retrieve the first spike time across all units

    Parameters
    ----------
    units: pynwb.misc.Units

    Returns
    -------

    """
    st = units['spike_times']
    inds = [0] + list(st.data[:-1])
    first_spikes = st.target.data[inds]
    return np.min(first_spikes)


def get_max_spike_time(units: pynwb.misc.Units):
    """Efficiently retrieve the last spike time across all units

    Parameters
    ----------
    units: pynwb.misc.Units

    Returns
    -------

    """
    st = units['spike_times']
    inds = list(st.data[:] - 1)
    last_spikes = st.target.data[inds]
    return np.max(last_spikes)


def align_by_times(units: pynwb.misc.Units, index, starts, stops):
    """
    Args:
        units: pynwb.misc.Units
        index: int
        starts: array-like
        stops: array-like
    Returns:
        np.array
    """
    return np.array([get_spike_times(units, index, [a, b]) - a for a, b in zip(starts, stops)])


def align_by_trials(units: pynwb.misc.Units, index, start_label='start_time',
                    stop_label=None, before=0., after=1.):
    """
    Args:
        units
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
    trials = units.get_ancestor('NWBFile').trials
    return align_by_time_intervals(units, index, trials, start_label, stop_label, before, after)


def align_by_time_intervals(units: pynwb.misc.Units, index, intervals, start_label='start_time',
                            stop_label='stop_time', before=0., after=0., trials_select=None):
    """
    Args:
        units: time-aware neurodata_type
        index: int
        intervals: pynwb.epoch.TimeIntervals
        start_label: str
            default: 'start_time'
        stop_label: str
            default: 'stop_time'
        before: float
            time after start_label in secs (positive goes back in time)
        after: float
            time after stop_label in secs (positive goes forward in time)
        trials_select: array_like, optional
            sub-selects specific trials
    Returns:
        np.array(shape=(n_trials, n_time, ...))
    """
    if stop_label is None:
        stop_label = 'start_time'
    if trials_select is None:
        starts = np.array(intervals[start_label][:]) - before
        stops = np.array(intervals[stop_label][:]) + after
    else:
        starts = np.array(intervals[start_label][trials_select]) - before
        stops = np.array(intervals[stop_label][trials_select]) + after
    return [x - before for x in align_by_times(units, index, starts, stops)]
