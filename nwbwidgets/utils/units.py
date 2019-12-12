import numpy as np
import pynwb

from bisect import bisect_right, bisect_left


def get_spike_times(units: pynwb.misc.Units, index, in_interval):
    """Use bisect methods to efficiently retrieve spikes from a given unit in a given interval

    Parameters
    ----------
    units: pynwb.misc.Units
    index: int
    in_interval: list of ints

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
