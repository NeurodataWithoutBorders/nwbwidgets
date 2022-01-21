import numpy as np
import pynwb

from bisect import bisect_right, bisect_left
from numpy import searchsorted


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
    st = units["spike_times"]
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
    st = units["spike_times"]
    inds = [0] + list(st.data[:-1])
    first_spikes = [st.target.data[i] for i in inds]
    return np.min(first_spikes)


def get_max_spike_time(units: pynwb.misc.Units):
    """Efficiently retrieve the last spike time across all units

    Parameters
    ----------
    units: pynwb.misc.Units

    Returns
    -------

    """
    st = units["spike_times"]
    inds = [x - 1 for x in st.data[:]]
    last_spikes = [st.target.data[i] for i in inds]
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

    st = units["spike_times"]
    unit_spike_data = st[index]

    istarts = searchsorted(unit_spike_data, starts)
    istops = searchsorted(unit_spike_data, stops)
    for start, istart, istop in zip(starts, istarts, istops):
        yield unit_spike_data[istart:istop] - start


def align_by_trials(
    units: pynwb.misc.Units,
    index,
    start_label="start_time",
    stop_label=None,
    start=-0.5,
    end=1.0,
):
    """
    Args:
        units
        start_label: str
            default: 'start_time'
        stop_label: str
            default: None (just align to start_time)
        start: float
            Start time for calculation before or after (negative or positive) the reference point (aligned to).
        end: float
            End time for calculation before or after (negative or positive) the reference point (aligned to).
    Returns:
        np.array(shape=(n_trials, n_time, ...))
    """
    trials = units.get_ancestor("NWBFile").trials
    return align_by_time_intervals(
        units, index, trials, start_label, stop_label, start, end
    )


def align_by_time_intervals(
    units: pynwb.misc.Units,
    index,
    intervals,
    start_label="start_time",
    stop_label=None,
    start=0.0,
    end=0.0,
    rows_select=(),
    progress_bar=None,
):
    """
    Args:
        units: time-aware neurodata_type
        index: int
        intervals: pynwb.epoch.TimeIntervals
        start_label: str
            default: 'start_time'
        stop_label: str
            default: 'stop_time'
        start: float
            Start time for calculation before or after (negative or positive) the reference point (aligned to).
        end: float
            End time for calculation before or after (negative or positive) the reference point (aligned to).
        rows_select: array_like, optional
            sub-selects specific rows
        progress_bar: FloatProgress, optional
            Proved a progress bar object to have this method automatically update the progress bar
    Returns:
        np.array(shape=(n_trials, n_time, ...))
    """
    if stop_label is None:
        stop_label = start_label
    starts = np.squeeze(np.array(intervals[start_label][:])[rows_select] + start)
    stops = np.squeeze(np.array(intervals[stop_label][:])[rows_select] + end)
    if progress_bar is not None:
        progress_bar.value = 0
        progress_bar.description = "reading spike data"

    out = []
    for i, x in enumerate(align_by_times(units, index, starts, stops)):
        out.append(x + start)
        if progress_bar is not None:
            progress_bar.value = i / len(units)

    return out


def get_unobserved_intervals(units, time_window, units_select=()):

    if "obs_intervals" not in units:
        return []

    # add observation intervals
    unobserved_intervals_list = []
    for i_unit in units_select:
        intervals = units["obs_intervals"][i_unit]  # TODO: use bisect here
        intervals = np.array(intervals, dtype="object")
        these_obs_intervals = intervals[
            (intervals[:, 1] > time_window[0]) & (intervals[:, 0] < time_window[1])
        ]
        unobs_intervals = np.c_[these_obs_intervals[:-1, 1], these_obs_intervals[1:, 0]]

        if len(these_obs_intervals):
            # handle unobserved interval on lower bound of window
            if these_obs_intervals[0, 0] > time_window[0]:
                unobs_intervals = np.vstack(
                    ([time_window[0], these_obs_intervals[0, 0]], unobs_intervals)
                )

            # handle unobserved interval on lower bound of window
            if these_obs_intervals[-1, 1] < time_window[1]:
                unobs_intervals = np.vstack(
                    (unobs_intervals, [these_obs_intervals[-1, 1], time_window[1]])
                )
        else:
            unobs_intervals = [time_window]

        unobserved_intervals_list.append(unobs_intervals)

    return unobserved_intervals_list
