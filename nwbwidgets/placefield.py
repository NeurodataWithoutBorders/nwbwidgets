import numpy as np
from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter, maximum_filter

import pynwb
from pynwb.misc import Units
from .behavior import plotly_show_spatial_trace
from .base import vis2widget
from ipywidgets import widgets
from .utils.units import get_spike_times

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

    # [] Work in buttons / dropdowns / sliders to modify following parameters in place field calculation:
        # [] Different epochs
        # [] Gaussian SD
        # [] Speed threshold
        # [] Minimum firing rate
        # [] Place field thresh (% of local max)

# Put widget rendering here
class PlaceFieldWidget(widgets.HBox):

    def __init__(self, unit: Units, node: pynwb.behavior.SpatialSeries):

        super().__init__()

        # Initialize receptive fields
        self.receptive_fields = np.zeros(firing_rate.shape, dtype=int)
        # Get pos
        pos, unit = get_timeseries_in_units(node)
        # Get time
        pos_tt = get_timeseries_tt(node)
        # Get spikes
        spikes = get_spike_times(unit)
        # Pixel width?
        pixel_width =

        # Put widget controls here:
            # - Gaussian SD
            # - Speed threshold
            # - Minimum firing rate
            # - Place field thresh (% of local max)

        self.compute_2d_firing_rate(pos, pos_tt, spikes, pixel_width) # speed_thresh=0.03, gaussian_sd=0.0184,
                                    # x_start=None, x_stop=None, y_start=None, y_stop=None)

        self.compute_2d_place_fields() # min_firing_rate=1, thresh=0.2,
                                                    # min_size=100):


        return vis2widget(plotly_show_spatial_trace(self.receptive_fields))


    # Put place field code here
    def find_nearest(self, arr, tt):
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

    def smooth(self, y, box_pts):
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

    def compute_speed(self, pos, pos_tt, smooth_param=40):
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
        return self.smooth(speed, smooth_param)


    def compute_2d_occupancy(self, pos, pos_tt, edges_x, edges_y, speed_thresh=0.03):
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
        is_running = self.compute_speed(pos, pos_tt) > speed_thresh
        run_pos = pos[is_running, :]
        occupancy = np.histogram2d(run_pos[:, 0],
                                   run_pos[:, 1],
                                   [edges_x, edges_y])[0] * sampling_period  # in seconds

        return occupancy, is_running


    def compute_2d_n_spikes(self, pos, pos_tt, spikes, edges_x, edges_y, speed_thresh=0.03):
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

        is_running = self.compute_speed(pos, pos_tt) > speed_thresh

        spike_pos_inds = self.find_nearest(spikes, pos_tt)
        spike_pos_inds = spike_pos_inds[is_running[spike_pos_inds]]
        pos_on_spikes = pos[spike_pos_inds, :]

        n_spikes = np.histogram2d(pos_on_spikes[:, 0],
                                  pos_on_spikes[:, 1],
                                  [edges_x, edges_y])[0]

        return n_spikes


    def compute_2d_firing_rate(self, pos, pos_tt, spikes,
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
        # pixel_width=0.0092,
        # field_len = 0.46
        # edges = np.arange(0, field_len + pixel_width, pixel_width)

        x_start = np.min(pos[:, 0]) if x_start is None else x_start
        x_stop = np.max(pos[:, 0]) if x_stop is None else x_stop

        y_start = np.min(pos[:, 1]) if y_start is None else y_start
        y_stop = np.max(pos[:, 1]) if y_stop is None else y_stop

        self.edges_x = np.arange(x_start, x_stop, pixel_width)
        self.edges_y = np.arange(y_start, y_stop, pixel_width)

        self.occupancy, running = self.compute_2d_occupancy(pos, pos_tt, self.edges_x, self.edges_y, speed_thresh)

        n_spikes = self.compute_2d_n_spikes(pos, pos_tt, spikes, self.edges_x, self.edges_y, speed_thresh)

        firing_rate = n_spikes / self.occupancy  # in Hz
        firing_rate[np.isnan(firing_rate)] = 0

        self.firing_rate = gaussian_filter(firing_rate, gaussian_sd / pixel_width)

        # return occupancy, filtered_firing_rate, [edges_x, edges_y]


    def compute_2d_place_fields(self, min_firing_rate=1, thresh=0.2,
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
        firing_rate = self.firing_rate
        local_maxima_inds = firing_rate == maximum_filter(firing_rate, 3)
        n_receptive_fields = 0
        firing_rate = firing_rate.copy()
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
                    self.receptive_fields[image_label] = n_receptive_fields
                    firing_rate[image_label] = 0

        # return receptive_fields