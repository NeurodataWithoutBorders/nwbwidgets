import numpy as np
from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter, maximum_filter

import matplotlib.pyplot as plt

import pynwb
from ipywidgets import widgets, BoundedFloatText, Dropdown

from .utils.widgets import interactive_output
from .utils.units import get_spike_times
from .utils.timeseries import get_timeseries_in_units, get_timeseries_tt
from .base import vis2widget

import plotly.graph_objects as go


## To-do
# [] Create PlaceFieldWidget class
# [X] Refactor place field calculation code to deal with nwb data type
# [X] Incorporate place field fxns into class
# [X] Change all internal attributes references
# [X]Change all internal method references

# [X] Get pos
# [X] Get time
# [X] Get spikes
# [] Get trials / epochs

# [] Submit draft PR

# [] Modify plotly_show_spatial_trace to plot 2D heatmap representing place fields or create new figure function?
# [] Dropdown that controls which unit

# [x] Work in buttons / dropdowns / sliders to modify following parameters in place field calculation:
# [] Different epochs
# [x] Gaussian SD
# [x] Speed threshold
# [] Minimum firing rate
# [] Place field thresh (% of local max)

# Put widget rendering here
def find_nearest(arr, tt):
    """Used for picking out elements of a TimeSeries based on spike times

    Parameters
    ----------
    arr
    tt

    Returns
    -------

    """
    arr = arr[arr > tt[0]]
    arr = arr[arr < tt[-1]]
    return np.searchsorted(tt, arr)


def smooth(y, box_pts):
    """Moving average

    Parameters
    ----------
    y
    box_pts

    Returns
    -------

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
    speed = np.hstack((0, np.sqrt(np.sum(np.diff(pos.T) ** 2, axis=0)) / np.diff(pos_tt)))
    return smooth(speed, smooth_param)


def compute_2d_occupancy(pos, pos_tt, edges_x, edges_y, speed_thresh=0.03):
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
    speed_thresh: float, optional
        in meters. Default = 3.0 cm/s

    Returns
    -------
    occupancy: np.ndarray(dtype=float)
        in seconds
    running: np.ndarray(dtype=bool)

    """

    sampling_period = (np.max(pos_tt) - np.min(pos_tt)) / len(pos_tt)
    is_running = compute_speed(pos, pos_tt) > speed_thresh
    run_pos = pos[is_running, :]
    occupancy = np.histogram2d(run_pos[:, 0],
                               run_pos[:, 1],
                               [edges_x, edges_y])[0] * sampling_period  # in seconds

    return occupancy, is_running


def compute_2d_n_spikes(pos, pos_tt, spikes, edges_x, edges_y, speed_thresh=0.03):
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
    speed_thresh: float
        in meters. Default = 3.0 cm/s

    Returns
    -------
    """

    is_running = compute_speed(pos, pos_tt) > speed_thresh

    spike_pos_inds = find_nearest(spikes, pos_tt)
    spike_pos_inds = spike_pos_inds[is_running[spike_pos_inds]]
    pos_on_spikes = pos[spike_pos_inds, :]

    n_spikes = np.histogram2d(pos_on_spikes[:, 0],
                              pos_on_spikes[:, 1],
                              [edges_x, edges_y])[0]

    return n_spikes


def compute_2d_firing_rate(pos, pos_tt, spikes,
                           pixel_width,
                           speed_thresh=0.03,
                           gaussian_sd=0.0184,
                           x_start=None, x_stop=None,
                           y_start=None, y_stop=None):
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
    pixel_width: float
    speed_thresh: float, optional
        in meters. Default = 3.0 cm/s
    gaussian_sd: float, optional
        in meters. Default = 1.84 cm
    x_start: float, optional
    x_stop: float, optional
    y_start: float, optional
    y_stop: float, optional


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

    edges_x = np.arange(x_start, x_stop, pixel_width)
    edges_y = np.arange(y_start, y_stop, pixel_width)

    occupancy, running = compute_2d_occupancy(pos, pos_tt, edges_x, edges_y, speed_thresh)

    n_spikes = compute_2d_n_spikes(pos, pos_tt, spikes, edges_x, edges_y, speed_thresh)

    firing_rate = n_spikes / occupancy  # in Hz
    firing_rate[np.isnan(firing_rate)] = 0  # get rid of NaNs so convolution works

    filtered_firing_rate = gaussian_filter(firing_rate, gaussian_sd / pixel_width)

    # filter occupancy to create a mask so non-explored regions are nan'ed
    filtered_occupancy = gaussian_filter(occupancy, gaussian_sd / pixel_width / 8)
    filtered_firing_rate[filtered_occupancy.astype('bool') < .00001] = np.nan

    return occupancy, filtered_firing_rate, [edges_x, edges_y]


def compute_2d_place_fields(firing_rate, min_firing_rate=1, thresh=0.2,
                            min_size=100):
    """Compute place fields

    Parameters
    ----------
    firing_rate: np.ndarray(NxN, dtype=float)
    min_firing_rate: float
        in Hz
    thresh: float
        % of local max
    min_size: float
        minimum size of place field in pixels

    Returns
    -------
    receptive_fields: np.ndarray(NxN, dtype=int)
        Each receptive field is labeled with a unique integer
    """
    local_maxima_inds = firing_rate == maximum_filter(firing_rate, 3)
    n_receptive_fields = 0
    firing_rate = firing_rate.copy()
    receptive_fields = {}
    for local_max in np.flipud(np.sort(firing_rate[local_maxima_inds])):
        labeled_image, num_labels = label(firing_rate > max(local_max * thresh,
                                                            min_firing_rate))
        if not num_labels:  # nothing above min_firing_thresh
            return
        for i in range(1, num_labels + 1):
            image_label = labeled_image == i
            if local_max in firing_rate[image_label]:
                break
            if np.sum(image_label) >= min_size:
                n_receptive_fields += 1
                receptive_fields[image_label] = n_receptive_fields
                firing_rate[image_label] = 0

    return receptive_fields


class PlaceFieldWidget(widgets.HBox):

    def __init__(self, spatial_series: pynwb.behavior.SpatialSeries, **kwargs):
        super().__init__()

        self.units = spatial_series.get_ancestor('NWBFile').units
        self.pos_tt = get_timeseries_tt(spatial_series)

        istart = 0
        istop = None
        self.pos, self.unit = get_timeseries_in_units(spatial_series, istart, istop)

        self.pixel_width = (np.nanmax(self.pos) - np.nanmin(self.pos)) / 1000

        # Put widget controls here:
        # - Minimum firing rate
        # - Place field thresh (% of local max)

        bft_gaussian = BoundedFloatText(value=0.0184, min=0, max=99999, description='gaussian sd (cm)')
        bft_speed = BoundedFloatText(value=0.03, min=0, max=99999, description='speed threshold (cm/s)')
        dd_unit_select = Dropdown(options=np.arange(len(self.units)), description='unit')

        self.controls = dict(
            gaussian_sd=bft_gaussian,
            speed_thresh=bft_speed,
            index=dd_unit_select
        )

        out_fig = interactive_output(self.do_rate_map, self.controls)

        self.children = [
            widgets.VBox([
                bft_gaussian,
                bft_speed,
                dd_unit_select
            ]),
            vis2widget(out_fig)
        ]

    def do_rate_map(self, index=0, speed_thresh=0.03, gaussian_sd=0.0184):
        tmin = min(self.pos_tt)
        tmax = max(self.pos_tt)

        spikes = get_spike_times(self.units, index, [tmin, tmax])

        occupancy, filtered_firing_rate, [edges_x, edges_y] = compute_2d_firing_rate(
            self.pos, self.pos_tt, spikes, self.pixel_width, speed_thresh=speed_thresh, gaussian_sd=gaussian_sd)

        fig, ax = plt.subplots()

        im = ax.imshow(filtered_firing_rate,
                       extent=[edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]],
                       aspect='equal')
        ax.set_xlabel('x ({})'.format(self.unit))
        ax.set_ylabel('y ({})'.format(self.unit))

        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('firing rate (Hz)')

        return fig


def compute_1d_occupancy(pos, spatial_bins, sampling_rate):
    finite_lin_pos = pos[np.isfinite(pos)]

    occupancy = np.histogram(
        finite_lin_pos, bins=spatial_bins)[0][:-2] / sampling_rate

    return occupancy


def compute_linear_firing_rate(pos, pos_tt, spikes, gaussian_sd=0.0557,
                               spatial_bin_len=0.0168):
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

    occupancy = compute_1d_occupancy(pos, spatial_bins, sampling_rate)

    # find pos_tt bin associated with each spike
    spike_pos_inds = find_nearest(spikes, pos_tt)

    pos_on_spikes = pos[spike_pos_inds]
    finite_pos_on_spikes = pos_on_spikes[np.isfinite(pos_on_spikes)]

    n_spikes = np.histogram(finite_pos_on_spikes, bins=spatial_bins)[0][:-2]

    firing_rate = np.nan_to_num(n_spikes / occupancy)

    filtered_firing_rate = gaussian_filter(
        firing_rate, gaussian_sd / spatial_bin_len)
    xx = spatial_bins[:-3] + (spatial_bins[1] - spatial_bins[0]) / 2

    return xx, occupancy, filtered_firing_rate


class PlaceField_1D_Widget(widgets.HBox):

    def __init__(self, spatial_series: pynwb.behavior.SpatialSeries, **kwargs):
        super().__init__()

        self.units = spatial_series.get_ancestor('NWBFile').units
        self.pos_tt = get_timeseries_tt(spatial_series)

        istart = 0
        istop = None
        self.pos, self.unit = get_timeseries_in_units(spatial_series, istart, istop)

        self.pixel_width = (np.nanmax(self.pos) - np.nanmin(self.pos)) / 1000

        # Put widget controls here:
        # - Minimum firing rate
        # - Place field thresh (% of local max)

        bft_gaussian = BoundedFloatText(value=0.0557, min=0, max=99999, description='gaussian sd (m)')
        bft_spatial_bin_len = BoundedFloatText(value=0.0168, min=0, max=99999, description='spatial bin length (m)')
        dd_unit_select = Dropdown(options=np.arange(len(self.units)), description='unit')

        self.controls = dict(
            gaussian_sd=bft_gaussian,
            spatial_bin_len=bft_spatial_bin_len,
            index=dd_unit_select
        )

        out_fig = interactive_output(self.do_1d_rate_map, self.controls)

        self.children = [
            widgets.VBox([
                bft_gaussian,
                bft_spatial_bin_len,
                dd_unit_select
            ]),
            vis2widget(out_fig)
        ]

    def do_1d_rate_map(self, index=0, gaussian_sd=0.0557, spatial_bin_len=0.0168):
        tmin = min(self.pos_tt)
        tmax = max(self.pos_tt)

        spikes = get_spike_times(self.units, index, [tmin, tmax])

        xx, occupancy, filtered_firing_rate = compute_linear_firing_rate(
            self.pos, self.pos_tt, spikes, gaussian_sd=gaussian_sd, spatial_bin_len=spatial_bin_len)

        fig, ax = plt.subplots()

        fig = ax.plot(xx, filtered_firing_rate, '-')
        ax.set_xlabel('x ({})'.format(self.unit))
        ax.set_ylabel('firing rate (Hz)')


        return fig
