import numpy as np
from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter, maximum_filter

import matplotlib.pyplot as plt
import nelpy.plotting as nlp

import pynwb
from ipywidgets import widgets, BoundedFloatText, Dropdown


from .utils.widgets import interactive_output
from .utils.units import get_spike_times
from .utils.timeseries import get_timeseries_in_units, get_timeseries_tt
from .base import vis2widget


## To-do
# [X] Create PlaceFieldWidget class
    # [X] Refactor place field calculation code to deal with nwb data type
        # [X] Incorporate place field fxns into class
        # [X] Change all internal attributes references
        # [X]Change all internal method references

    # [X] Get pos
    # [X] Get time
    # [X] Get spikes
    # [] Get trials / epochs

# [X] Submit draft PR

    # [] 1D Place Field Widget
        # [X] Incorporate nelpy package into widget
        # [] Add foreign group and sort controller to pick unit groups and ranges?
        # [] Normalized firing rate figure?
        # [] Add collapsed unit vizualization?
        # [] Scale bar?
        # [] Sort place cell tuning curves by peak firing rate position?
        # [] Color palette control?

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


def compute_1d_occupancy(pos, pos_tt, spatial_bins, sampling_rate, speed_thresh=0.03):

    is_running = compute_speed(pos, pos_tt) > speed_thresh
    run_pos = pos[is_running, :]
    finite_lin_pos = run_pos[np.isfinite(run_pos)]

    occupancy = np.histogram(
        finite_lin_pos, bins=spatial_bins)[0][:-2] / sampling_rate

    return occupancy


def compute_linear_firing_rate(pos, pos_tt, spikes, gaussian_sd=0.0557,
                               spatial_bin_len=0.0168, speed_thresh=0.03):
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

    occupancy = compute_1d_occupancy(pos, pos_tt, spatial_bins, sampling_rate)

    is_running = compute_speed(pos, pos_tt) > speed_thresh

    # find pos_tt bin associated with each spike
    spike_pos_inds = find_nearest(spikes, pos_tt)
    spike_pos_inds = spike_pos_inds[is_running[spike_pos_inds]]
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
                 # foreign_group_and_sort_controller: GroupAndSortController = None,
                 # group_by=None,

        super().__init__()

        # if foreign_group_and_sort_controller:
        #     self.gas = foreign_group_and_sort_controller
        # else:
        #     self.gas = self.make_group_and_sort(group_by=group_by, control_order=False)

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
            # index=dd_unit_select
        )

        out_fig = interactive_output(self.do_1d_rate_map, self.controls)

        self.children = [
            widgets.VBox([
                bft_gaussian,
                bft_spatial_bin_len,
                # dd_unit_select
            ]),
            vis2widget(out_fig)
        ]

    def do_1d_rate_map(self, gaussian_sd=0.0557, spatial_bin_len=0.0168):
        tmin = min(self.pos_tt)
        tmax = max(self.pos_tt)

        index = np.arange(len(self.units))

        spikes = get_spike_times(self.units, index[0], [tmin, tmax])
        xx, occupancy, filtered_firing_rate = compute_linear_firing_rate(
            self.pos, self.pos_tt, spikes, gaussian_sd=gaussian_sd, spatial_bin_len=spatial_bin_len)

        all_unit_firing_rate = np.zeros([len(self.units), len(xx)])
        all_unit_firing_rate[0] = filtered_firing_rate

        for ind in index[1:]:
            spikes = get_spike_times(self.units, ind, [tmin, tmax])
            _, _, all_unit_firing_rate[ind] = compute_linear_firing_rate(
                self.pos, self.pos_tt, spikes, gaussian_sd=gaussian_sd, spatial_bin_len=spatial_bin_len)

        # npl.set_palette(npl.colors.rainbow)
        # with npl.FigureManager(show=True, figsize=(8, 8)) as (fig, ax):
        #     npl.utils.skip_if_no_output(fig)
        fig, ax = plt.subplots()
        plot_tuning_curves1D(all_unit_firing_rate, xx, ax=ax, unit_labels=index)

        # fig = ax.plot(xx, filtered_firing_rate, '-')
        # ax.set_xlabel('x ({})'.format(self.unit))
        # ax.set_ylabel('firing rate (Hz)')

        return fig

def plot_tuning_curves1D(ratemap, bin_pos, ax=None, normalize=False, pad=10, unit_labels=None, fill=True, color=None):
    """
    WARNING! This function is not complete, and hence 'private',
    and may be moved somewhere else later on.

    If pad=0 then the y-axis is assumed to be firing rate
    """
    xmin = bin_pos[0]
    xmax = bin_pos[-1]
    xvals = bin_pos

    n_units, n_ext = ratemap.shape

    # if normalize:
    #     peak_firing_rates = ratemap.max(axis=1)
    #     ratemap = (ratemap.T / peak_firing_rates).T

    # determine max firing rate
    max_firing_rate = ratemap.max()

    if xvals is None:
        xvals = np.arange(n_ext)
    if xmin is None:
        xmin = xvals[0]
    if xmax is None:
        xmax = xvals[-1]

    for unit, curve in enumerate(ratemap):
        if color is None:
            line = ax.plot(xvals, unit*pad + curve, zorder=int(10+2*n_units-2*unit))
        else:
            line = ax.plot(xvals, unit*pad + curve, zorder=int(10+2*n_units-2*unit), color=color)
        if fill:
            # Get the color from the current curve
            fillcolor = line[0].get_color()
            ax.fill_between(xvals, unit*pad, unit*pad + curve, alpha=0.3, color=fillcolor, zorder=int(10+2*n_units-2*unit-1))

    ax.set_xlim(xmin, xmax)
    if pad != 0:
        yticks = np.arange(n_units)*pad + 0.5*pad
        ax.set_yticks(yticks)
        ax.set_yticklabels(unit_labels)
        ax.set_xlabel('external variable')
        ax.set_ylabel('unit')
        nlp.utils.no_yticks(ax)
        nlp.utils.clear_left(ax)
    else:
        if normalize:
            ax.set_ylabel('normalized firing rate')
        else:
            ax.set_ylabel('firing rate [Hz]')
        ax.set_ylim(0)

    nlp.utils.clear_top(ax)
    nlp.utils.clear_right(ax)

    return ax
