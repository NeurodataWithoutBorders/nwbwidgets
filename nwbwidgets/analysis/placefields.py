from functools import lru_cache

import numpy as np
from scipy.ndimage.filters import gaussian_filter


def find_nearest(arr, tt):
    """Used for picking out elements of a TimeSeries based on spike times

    Parameters
    ----------
    arr: array-like
    tt: array-like

    Returns
    -------
    indices: array-like

    """
    arr = arr[arr > tt[0]]
    arr = arr[arr < tt[-1]]
    return np.searchsorted(tt, arr)


def smooth(y, box_pts):
    """Moving average

    Parameters
    ----------
    y: array-like
    box_pts: int

    Returns
    -------
    output: np.array(dtype=float)

    """
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')


def compute_speed(pos, pos_tt, smooth_param=40):
    """Compute boolean of whether the speed of the animal was above a threshold
    for each time point

    Parameters
    ----------
    pos: np.ndarray(dtype=float)
        in meters
    pos_tt: np.ndarray(dtype=float)
        in seconds
    smooth_param: float, optional

    Returns
    -------
    running: np.ndarray(dtype=bool)

    """
    if len(pos.shape) > 1:
        speed = np.hstack((0, np.sqrt(np.sum(np.diff(pos.T) ** 2, axis=0)) / np.diff(pos_tt)))
    else:
        speed = np.hstack((0, np.sqrt(np.diff(pos.T) ** 2) / np.diff(pos_tt)))
    return smooth(speed, smooth_param)


def compute_2d_occupancy(pos, pos_tt, edges_x, edges_y, pixel_width, speed_thresh=0.03, velocity=None):
    """Computes occupancy per bin in seconds

    Parameters
    ----------
    pos: np.ndarray(dtype=float)
        in meters
    pos_tt: np.ndarray(dtype=float)
        in seconds
    edges_x: array-like
        edges of histogram in meters
    edges_y: array-like
        edges of histogram in meters
    pixel_width: array-like
    speed_thresh: float, optional
        in meters. Default = 3.0 cm/s
    velocity: np.ndarray(dtype=float)
        pre-computed velocity

    Returns
    -------
    occupancy: np.ndarray(dtype=float)
        in seconds
    running: np.ndarray(dtype=bool)

    """

    sampling_period = (np.max(pos_tt) - np.min(pos_tt)) / len(pos_tt)
    np.seterr(invalid='ignore')
    if velocity is None:
        if pixel_width[1] is not int(1):
            is_running = compute_speed(pos, pos_tt) > speed_thresh
        else:
            is_running = compute_speed(pos[:, 0], pos_tt) > speed_thresh
    else:
        is_running = np.linalg.norm(velocity) > speed_thresh

    run_pos = pos[is_running, :]
    occupancy = np.histogram2d(run_pos[:, 1],
                               run_pos[:, 0],
                               [edges_y, edges_x])[0] * sampling_period  # in seconds

    return occupancy, is_running


def compute_2d_n_spikes(pos, pos_tt, spikes, edges_x, edges_y, pixel_width, speed_thresh=0.03, velocity=None):
    """Returns speed-gated position during spikes

    Parameters
    ----------
    pos: np.ndarray(dtype=float)
        (time x 2) in meters
    pos_tt: np.ndarray(dtype=float)
        (time,) in seconds
    spikes: np.ndarray(dtype=float)
        in seconds
    edges_x: np.ndarray(dtype=float)
        edges of histogram in meters
    edges_y: np.ndarray(dtype=float)
        edges of histogram in meters
    pixel_width: array
    speed_thresh: float
        in meters. Default = 3.0 cm/s
    velocity: np.ndarray(dtype=float)
        pre-computed velocity

    Returns
    -------
    """
    np.seterr(invalid='ignore')
    if velocity is None:
        if pixel_width[1] is not int(1):
            is_running = compute_speed(pos, pos_tt) > speed_thresh
        else:
            is_running = compute_speed(pos[:, 0], pos_tt) > speed_thresh
    else:
        is_running = np.linalg.norm(velocity) > speed_thresh

    spike_pos_inds = find_nearest(spikes, pos_tt)
    spike_pos_inds = spike_pos_inds[is_running[spike_pos_inds]]
    pos_on_spikes = pos[spike_pos_inds, :]

    n_spikes = np.histogram2d(pos_on_spikes[:, 1],
                              pos_on_spikes[:, 0],
                              [edges_y, edges_x])[0]

    return n_spikes


def compute_2d_firing_rate(pos, pos_tt, spikes,
                           pixel_width,
                           speed_thresh=0.03,
                           gaussian_sd_x=0.0184,
                           gaussian_sd_y=0.0184,
                           x_start=None, x_stop=None,
                           y_start=None, y_stop=None,
                           velocity=None):
    """Returns speed-gated occupancy and speed-gated and
    Gaussian-filtered firing rate

    Parameters
    ----------
    pos: np.ndarray(dtype=float)
        (time x 2), in meters
    pos_tt: np.ndarray(dtype=float)
        (time,) in seconds
    spikes: np.ndarray(dtype=float)
        in seconds
    pixel_width: array-like
    speed_thresh: float, optional
        in meters. Default = 3.0 cm/s
    gaussian_sd_x: float, optional
        width of gaussian kernel in x-dim, in meters. Default = 1.84 cm
    gaussian_sd_y: float, optional
        width of gaussian kernel in y-dim, in meters. Default = 1.84 cm
    x_start: float, optional
    x_stop: float, optional
    y_start: float, optional
    y_stop: float, optional
    velocity: np.ndarray(dtype=float)
        pre-computed velocity

    Returns
    -------

    occupancy: np.ndarray
        in seconds
    filtered_firing_rate: np.ndarray(shape=(cell, x, y), dtype=float)
        in Hz

    """

    x_start = np.nanmin(pos[:, 0]) if x_start is None else x_start
    x_stop = np.nanmax(pos[:, 0]) if x_stop is None else x_stop

    y_start = np.nanmin(pos[:, 1]) if y_start is None else y_start
    y_stop = np.nanmax(pos[:, 1]) if y_stop is None else y_stop

    edges_x = np.arange(x_start, x_stop, pixel_width[0])
    edges_y = np.arange(y_start, y_stop, pixel_width[1])

    occupancy, running = compute_2d_occupancy(pos, pos_tt, edges_x, edges_y, pixel_width, speed_thresh, velocity)

    n_spikes = compute_2d_n_spikes(pos, pos_tt, spikes, edges_x, edges_y, pixel_width, speed_thresh, velocity)

    np.seterr(divide='ignore')
    firing_rate = n_spikes / occupancy  # in Hz
    firing_rate[np.isnan(firing_rate)] = 0  # get rid of NaNs so convolution works
    sigmas = [gaussian_sd_y / pixel_width[1], gaussian_sd_x / pixel_width[0]]
    filtered_firing_rate = gaussian_filter(firing_rate, sigmas)

    # filter occupancy to create a mask so non-explored regions are nan'ed
    sigmas_occ = [gaussian_sd_y / pixel_width[1] / 8, gaussian_sd_x / pixel_width[0] / 8]
    filtered_occupancy = gaussian_filter(occupancy, sigmas_occ)
    filtered_firing_rate[filtered_occupancy.astype('bool') < .00001] = np.nan

    return occupancy, filtered_firing_rate, [edges_x, edges_y]


def compute_1d_occupancy(pos, pos_tt, spatial_bins, sampling_rate, speed_thresh=0.03, velocity=None):

    np.seterr(invalid='ignore')
    if velocity is None:
        is_running = compute_speed(pos, pos_tt) > speed_thresh
    else:
        is_running = np.linalg.norm(velocity) > speed_thresh

    run_pos = pos[is_running, :]
    finite_lin_pos = run_pos[np.isfinite(run_pos)]

    occupancy = np.histogram(
        finite_lin_pos, bins=spatial_bins)[0][:-2] / sampling_rate

    return occupancy


def compute_linear_firing_rate(pos, pos_tt, spikes, gaussian_sd=0.0557,
                               spatial_bin_len=0.0168, speed_thresh=0.03, velocity=None):
    """The occupancy and number of spikes, speed-gated, binned, and smoothed
    over position

    Parameters
    ----------
    pos: np.ndarray
        linearized position
    pos_tt: np.ndarray
        sample times in seconds
    spikes: np.ndarray
        for a single cell in seconds
    gaussian_sd: float (optional)
        in meters. Default = 5.57 cm
    spatial_bin_len: float (optional)
        in meters. Default = 1.68 cm
    speed_thresh: float (optional)
        in m/s. Default = 0.03
    velocity: np.ndarray(dtype=float)
        pre-computed velocity

    Returns
    -------
    xx: np.ndarray
        center of position bins in meters
    occupancy: np.ndarray
        time in each spatial bin in seconds, during appropriate trials and
        while running
    filtered_n_spikes: np.ndarray
        number of spikes in each spatial bin,  during appropriate trials, while
        running, and processed with a Gaussian filter

    """
    spatial_bins = np.arange(np.nanmin(pos), np.nanmax(pos) + spatial_bin_len, spatial_bin_len)

    sampling_rate = len(pos_tt) / (np.nanmax(pos_tt) - np.nanmin(pos_tt))

    occupancy = compute_1d_occupancy(pos, pos_tt, spatial_bins, sampling_rate, speed_thresh, velocity)

    np.seterr(invalid='ignore')
    is_running = compute_speed(pos, pos_tt) > speed_thresh

    # find pos_tt bin associated with each spike
    spike_pos_inds = find_nearest(spikes, pos_tt)
    spike_pos_inds = spike_pos_inds[is_running[spike_pos_inds]]
    pos_on_spikes = pos[spike_pos_inds]
    finite_pos_on_spikes = pos_on_spikes[np.isfinite(pos_on_spikes)]

    n_spikes = np.histogram(finite_pos_on_spikes, bins=spatial_bins)[0][:-2]

    np.seterr(divide='ignore')
    firing_rate = np.nan_to_num(n_spikes / occupancy)

    filtered_firing_rate = gaussian_filter(
        firing_rate, gaussian_sd / spatial_bin_len)
    xx = spatial_bins[:-3] + (spatial_bins[1] - spatial_bins[0]) / 2

    return xx, occupancy, filtered_firing_rate
