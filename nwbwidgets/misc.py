from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pynwb
import scipy
from ipywidgets import widgets, fixed, FloatProgress, Layout
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from pynwb.misc import AnnotationSeries, Units, DecompositionSeries

from .analysis.spikes import compute_smoothed_firing_rate
from .controllers import (
    make_trial_event_controller,
    GroupAndSortController,
    StartAndDurationController,
    ProgressBar,
)
from .utils.dynamictable import infer_categorical_columns, extract_data_from_intervals
from .utils.mpl import create_big_ax
from .utils.plotly import event_group
from .utils.units import (
    get_spike_times,
    get_max_spike_time,
    get_min_spike_time,
    align_by_time_intervals,
    get_unobserved_intervals,
)
from .utils.widgets import interactive_output


color_wheel = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def show_annotations(annotations: AnnotationSeries, **kwargs):
    fig, ax = plt.subplots()
    ax.eventplot(annotations.timestamps, **kwargs)
    ax.set_xlabel("time (s)")
    return fig


def show_session_raster(
    units: Units,
    time_window=None,
    units_window=None,
    show_obs_intervals=True,
    order=None,
    group_inds=None,
    labels=None,
    show_legend=True,
    progress_bar=None,
):
    """

    Parameters
    ----------
    units: pynwb.misc.Units
    time_window: [int, int]
    units_window: [int, int]
    show_obs_intervals: bool
    order: array-like, optional
    group_inds: array-like, optional
    labels: array-like, optional
    show_legend: bool
        default = True
        Does not show legend if color_by is None or 'id'.
    progress_bar: FloatProgress, optional

    Returns
    -------
    matplotlib.pyplot.Figure

    """

    if time_window is None:
        time_window = [get_min_spike_time(units), get_max_spike_time(units)]

    if units_window is None:
        units_window = [0, len(units)]

    if order is None:
        order = np.arange(len(units), dtype="int")

    if progress_bar:
        this_iter = ProgressBar(order, desc="reading spike data", leave=False)
        progress_bar = this_iter.container
    else:
        this_iter = order
    data = []
    for unit in this_iter:
        data.append(get_spike_times(units, unit, time_window))

    if show_obs_intervals:
        unobserved_intervals_list = get_unobserved_intervals(units, time_window, order)
    else:
        unobserved_intervals_list = None

    ax = plot_grouped_events(
        data,
        time_window,
        group_inds=group_inds,
        labels=labels,
        show_legend=show_legend,
        offset=units_window[0],
        unobserved_intervals_list=unobserved_intervals_list,
        progress_bar=progress_bar,
    )
    ax.set_ylabel("unit #")
    if len(data) <= 30:
        unit_id_display = np.array(units.id.data[:])[[x for x in this_iter]]
        ax.set_yticklabels(unit_id_display)
    else:
        ax.axes.yaxis.set_visible(False)

    return ax


class RasterWidget(widgets.HBox):
    def __init__(
        self,
        units: Units,
        foreign_time_window_controller: StartAndDurationController = None,
        foreign_group_and_sort_controller: GroupAndSortController = None,
        group_by=None,
    ):
        super().__init__()

        self.units = units

        if foreign_time_window_controller is None:
            self.tmin = get_min_spike_time(units)
            self.tmax = get_max_spike_time(units)
            self.time_window_controller = StartAndDurationController(
                tmin=self.tmin, tmax=self.tmax
            )
        else:
            self.time_window_controller = foreign_time_window_controller

        if foreign_group_and_sort_controller:
            self.gas = foreign_group_and_sort_controller
        else:
            self.gas = self.make_group_and_sort(group_by=group_by, control_order=False)

        self.progress_bar = widgets.HBox()

        self.controls = dict(time_window=self.time_window_controller, gas=self.gas)

        plot_func = partial(
            show_session_raster, units=self.units, progress_bar=self.progress_bar
        )

        out_fig = interactive_output(plot_func, self.controls)

        if foreign_time_window_controller:
            right_panel = widgets.VBox(
                children=[
                    self.progress_bar,
                    out_fig,
                ],
                layout=Layout(width="100%"),
            )
        else:
            right_panel = widgets.VBox(
                children=[
                    self.time_window_controller,
                    self.progress_bar,
                    out_fig,
                ],
                layout=Layout(width="100%"),
            )

        if foreign_group_and_sort_controller:
            self.children = [right_panel]
        else:
            self.children = [self.gas, right_panel]

        self.layout = Layout(width="100%")

    def make_group_and_sort(self, group_by=None, control_order=True):
        return GroupAndSortController(
            self.units, group_by=group_by, control_order=control_order
        )


def show_decomposition_series(node, **kwargs):
    # Use Rendering... as a placeholder
    ntabs = 2
    children = [widgets.HTML("Rendering...") for _ in range(ntabs)]

    def on_selected_index(change):
        # Click on Traces Tab
        if change.new == 1 and isinstance(change.owner.children[1], widgets.HTML):
            widget_box = show_decomposition_traces(node)
            children[1] = widget_box
            change.owner.children = children

    field_lay = widgets.Layout(
        max_height="40px", max_width="500px", min_height="30px", min_width="130px"
    )
    vbox = []
    for key, val in node.fields.items():
        lbl_key = widgets.Label(key + ":", layout=field_lay)
        lbl_val = widgets.Label(str(val), layout=field_lay)
        vbox.append(widgets.HBox(children=[lbl_key, lbl_val]))
    children[0] = widgets.VBox(vbox)

    tab_nest = widgets.Tab()
    tab_nest.children = children
    tab_nest.set_title(0, "Fields")
    tab_nest.set_title(1, "Traces")
    tab_nest.observe(on_selected_index, names="selected_index")
    return tab_nest


def show_decomposition_traces(node: DecompositionSeries):
    # Produce figure
    def control_plot(x0, x1, ch0, ch1):
        fig, ax = plt.subplots(nrows=nBands, ncols=1, sharex=True, figsize=(14, 7))
        for bd in range(nBands):
            data = node.data[x0:x1, ch0 : ch1 + 1, bd]
            xx = np.arange(x0, x1)
            mu_array = np.mean(data, 0)
            sd_array = np.std(data, 0)
            offset = np.mean(sd_array) * 5
            yticks = [i * offset for i in range(ch1 + 1 - ch0)]
            for i in range(ch1 + 1 - ch0):
                ax[bd].plot(xx, data[:, i] - mu_array[i] + yticks[i])
            ax[bd].set_ylabel("Ch #", fontsize=20)
            ax[bd].set_yticks(yticks)
            ax[bd].set_yticklabels([str(i) for i in range(ch0, ch1 + 1)])
            ax[bd].tick_params(axis="both", which="major", labelsize=16)
        ax[bd].set_xlabel("Time [ms]", fontsize=20)
        return fig

    nSamples = node.data.shape[0]
    nChannels = node.data.shape[1]
    nBands = node.data.shape[2]
    fs = node.rate

    # Controls
    field_lay = widgets.Layout(
        max_height="40px", max_width="100px", min_height="30px", min_width="70px"
    )
    x0 = widgets.BoundedIntText(
        value=0, min=0, max=int(1000 * nSamples / fs - 100), layout=field_lay
    )
    x1 = widgets.BoundedIntText(
        value=nSamples, min=100, max=int(1000 * nSamples / fs), layout=field_lay
    )
    ch0 = widgets.BoundedIntText(
        value=0, min=0, max=int(nChannels - 1), layout=field_lay
    )
    ch1 = widgets.BoundedIntText(
        value=10, min=0, max=int(nChannels - 1), layout=field_lay
    )

    controls = {"x0": x0, "x1": x1, "ch0": ch0, "ch1": ch1}
    out_fig = widgets.interactive_output(control_plot, controls)

    # Assemble layout box
    lbl_x = widgets.Label("Time [ms]:", layout=field_lay)
    lbl_ch = widgets.Label("Ch #:", layout=field_lay)
    lbl_blank = widgets.Label("    ", layout=field_lay)
    hbox0 = widgets.HBox(children=[lbl_x, x0, x1, lbl_blank, lbl_ch, ch0, ch1])
    vbox = widgets.VBox(children=[hbox0, out_fig])
    return vbox


class PSTHWidget(widgets.VBox):
    def __init__(
        self,
        input_data: Units,
        trials: pynwb.epoch.TimeIntervals = None,
        unit_index=0,
        unit_controller=None,
        ntt=1000,
    ):

        self.units = input_data

        super().__init__()

        if trials is None:
            self.trials = self.get_trials()
            if self.trials is None:
                self.children = [widgets.HTML("No trials present")]
                return
        else:
            self.trials = trials

        if unit_controller is None:
            self.unit_ids = self.units.id.data[:]
            n_units = len(self.unit_ids)
            self.unit_controller = widgets.Dropdown(
                options=[(str(self.unit_ids[x]), x) for x in range(n_units)],
                value=unit_index,
                description="unit",
                layout=Layout(width="200px"),
            )

        self.trial_event_controller = make_trial_event_controller(
            self.trials, layout=Layout(width="200px")
        )
        self.before_ft = widgets.FloatText(
            0.5, min=0, description="before (s)", layout=Layout(width="200px")
        )
        self.after_ft = widgets.FloatText(
            2.0, min=0, description="after (s)", layout=Layout(width="200px")
        )
        self.psth_type_radio = widgets.RadioButtons(
            options=["histogram", "gaussian"], layout=Layout(width="100px")
        )
        self.bins_ft = widgets.IntText(
            30, min=0, description="# bins", layout=Layout(width="150px")
        )
        self.gaussian_sd_ft = widgets.FloatText(
            0.05,
            min=0.001,
            description="sd (s)",
            layout=Layout(width="150px"),
            active=False,
            step=0.01,
        )

        self.gas = self.make_group_and_sort(window=False, control_order=False)

        self.controls = dict(
            ntt=fixed(ntt),
            index=self.unit_controller,
            after=self.after_ft,
            before=self.before_ft,
            start_label=self.trial_event_controller,
            gas=self.gas,
            plot_type=self.psth_type_radio,
            sigma_in_secs=self.gaussian_sd_ft,
            nbins=self.bins_ft
            # progress_bar=fixed(progress_bar)
        )

        out_fig = interactive_output(self.update, self.controls)

        self.children = [
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            self.gas,
                            widgets.HBox(
                                [
                                    self.psth_type_radio,
                                    widgets.VBox([self.gaussian_sd_ft, self.bins_ft]),
                                ]
                            ),
                        ]
                    ),
                    widgets.VBox(
                        [
                            self.unit_controller,
                            self.trial_event_controller,
                            self.before_ft,
                            self.after_ft,
                        ]
                    ),
                ]
            ),
            out_fig,
        ]

    def get_trials(self):
        return self.units.get_ancestor("NWBFile").trials

    def make_group_and_sort(self, window=None, control_order=False):
        return GroupAndSortController(
            self.trials, window=window, control_order=control_order
        )

    def update(
        self,
        index: int,
        start_label: str = "start_time",
        before: float = 0.0,
        after: float = 1.0,
        order=None,
        group_inds=None,
        labels=None,
        sigma_in_secs=0.05,
        ntt: int = 1000,
        progress_bar=None,
        figsize=(7, 7),
        nbins=30,
        plot_type="histogram",
        align_line_color=(0.7, 0.7, 0.7),
    ):
        """

        Parameters
        ----------
        index: int
            Index of unit
        start_label: str, optional
            Trial column name to align on
        before: float
            Time before that event (should be positive)
        after: float
            Time after that event
        order
        group_inds
        labels
        sigma_in_secs: float, optional
            standard deviation of gaussian kernel
        ntt:
            Number of time points to use for smooth curve
        progress_bar:
        figsize: tuple, optional

        Returns
        -------
        matplotlib.Figure

        """

        data = align_by_time_intervals(
            self.units,
            index,
            self.trials,
            start_label,
            start_label,
            before,
            after,
            order,
            progress_bar=progress_bar,
        )

        fig, axs = plt.subplots(2, 1, figsize=figsize)

        show_psth_raster(
            data,
            before,
            after,
            group_inds,
            labels,
            ax=axs[0],
            progress_bar=progress_bar,
        )

        axs[0].set_title(f"PSTH for unit {self.unit_ids[index]}")
        axs[0].set_xticks([])
        axs[0].set_xlabel("")

        if plot_type == "gaussian":
            self.bins_ft.layout.visibility = "hidden"
            self.bins_ft.layout.height = "0px"
            self.gaussian_sd_ft.layout.visibility = None
            self.gaussian_sd_ft.layout.height = None
            # expanded data so that gaussian smoother uses larger window than is viewed
            expanded_data = align_by_time_intervals(
                self.units,
                index,
                self.trials,
                start_label,
                start_label,
                before + sigma_in_secs * 4,
                after + sigma_in_secs * 4,
                order,
                progress_bar=progress_bar,
            )
            show_psth_smoothed(
                expanded_data,
                axs[1],
                before + sigma_in_secs * 4,
                after + sigma_in_secs * 4,
                group_inds,
                sigma_in_secs=sigma_in_secs,
                ntt=ntt,
            )
        elif plot_type == "histogram":
            self.gaussian_sd_ft.layout.visibility = "hidden"
            self.gaussian_sd_ft.layout.height = "0px"
            self.bins_ft.layout.visibility = None
            self.bins_ft.layout.height = None
            show_histogram(data, axs[1], before, after, group_inds, nbins=nbins)
        else:
            raise ValueError(
                "unsupported plot type {}".format(self.psth_type_radio.value)
            )

        axs[1].set_xlim([-before, after])
        axs[1].set_ylabel("firing rate (Hz)")
        axs[1].set_xlabel("time (s)")
        axs[1].axvline(color=align_line_color)

        return fig


def show_histogram(
    data, ax: plt.Axes, before: float, after: float, group_inds=None, nbins: int = 30
):
    if not len(data):
        return

    if group_inds is None:
        height, x = np.histogram(np.hstack(data), bins=nbins, range=(-before, after))
        width = np.diff(x[:2])
        height = height / len(data) / width
        plt.bar(x[:-1], height, edgecolor=(0.3, 0.3, 0.3), width=width, align="edge")
    else:
        data = np.asarray(data, dtype="object")
        # group_inds = np.asarray(group_inds)
        for group in np.unique(group_inds):
            this_data = np.hstack(data[group_inds == group])
            height, x = np.histogram(this_data, bins=nbins, range=(-before, after))
            width = np.diff(x[:2])
            height = height / np.sum(group_inds == group) / width
            ax.bar(
                x[:-1],
                height,
                color=color_wheel[group % len(color_wheel)],
                edgecolor=(0.3, 0.3, 0.3),
                width=width,
                align="edge",
                alpha=0.6,
            )


def show_psth_smoothed(
    data,
    ax,
    before: float,
    after: float,
    group_inds=None,
    sigma_in_secs: float = 0.05,
    ntt: int = 1000,
):
    if not len(data):  # TODO: when does this occur?
        return
    all_data = np.hstack(data)
    if not len(all_data):  # no spikes
        return
    tt = np.linspace(-before, after, ntt)
    smoothed = np.array(
        [compute_smoothed_firing_rate(x, tt, sigma_in_secs) for x in data]
    )

    if group_inds is None:
        group_inds = np.zeros((len(smoothed)), dtype=np.int)
    group_stats = []
    for group in np.unique(group_inds):
        this_mean = np.mean(smoothed[group_inds == group], axis=0)
        err = scipy.stats.sem(smoothed[group_inds == group], axis=0)
        group_stats.append(
            dict(
                mean=this_mean,
                lower=this_mean - 2 * err,
                upper=this_mean + 2 * err,
                group=group,
            )
        )
    for stats in group_stats:
        color = color_wheel[stats["group"]]
        ax.plot(tt, stats["mean"], color=color)
        ax.fill_between(tt, stats["lower"], stats["upper"], alpha=0.2, color=color)


def plot_grouped_events(
    data,
    window,
    group_inds=None,
    colors=color_wheel,
    ax=None,
    labels=None,
    show_legend=True,
    offset=0,
    unobserved_intervals_list=None,
    progress_bar=None,
    figsize=(8, 6),
):
    """

    Parameters
    ----------
    data: array-like
    window: array-like [float, float]
        Time in seconds
    group_inds: array-like dtype=int, optional
    colors: array-like, optional
    ax: plt.Axes, optional
    labels: array-like dtype=str, optional
    show_legend: bool, optional
    offset: number, optional
    unobserved_intervals_list: array-like, optional
    progress_bar: FloatProgress, optional
    figsize: tuple, optional

    Returns
    -------

    """

    data = np.asarray(data, dtype="object")
    legend_kwargs = dict()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        if hasattr(fig, "canvas"):
            fig.canvas.header_visible = False
        else:
            legend_kwargs.update(bbox_to_anchor=(1.01, 1))
    if group_inds is not None:
        ugroup_inds = np.unique(group_inds)
        handles = []

        if progress_bar is not None:
            this_iter = ProgressBar(
                enumerate(ugroup_inds),
                desc="plotting spikes",
                leave=False,
                total=len(ugroup_inds),
            )
            progress_bar = this_iter.container
        else:
            this_iter = enumerate(ugroup_inds)

        for i, ui in this_iter:
            color = colors[ugroup_inds[i] % len(colors)]
            lineoffsets = np.where(group_inds == ui)[0] + offset
            event_collection = ax.eventplot(
                data[group_inds == ui],
                orientation="horizontal",
                lineoffsets=lineoffsets,
                color=color,
            )
            handles.append(event_collection[0])
        if show_legend:
            ax.legend(
                handles=handles[::-1],
                labels=list(labels[ugroup_inds][::-1]),
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                **legend_kwargs,
            )
    else:
        ax.eventplot(
            data,
            orientation="horizontal",
            color="k",
            lineoffsets=np.arange(len(data)) + offset,
        )

    if unobserved_intervals_list is not None:
        plot_unobserved_intervals(unobserved_intervals_list, ax, offset=offset)

    ax.set_xlim(window)
    ax.set_xlabel("time (s)")
    ax.set_ylim(np.array([-0.5, len(data) - 0.5]) + offset)
    if len(data) <= 30:
        ax.set_yticks(range(offset, len(data) + offset))
        ax.set_yticklabels(range(offset, len(data) + offset))

    return ax


def plot_unobserved_intervals(
    unobserved_intervals_list, ax, offset=0, color=(0.85, 0.85, 0.85)
):
    for irow, unobs_intervals in enumerate(unobserved_intervals_list):
        rects = [
            Rectangle(
                (i_interval[0], irow - 0.5 + offset), i_interval[1] - i_interval[0], 1
            )
            for i_interval in unobs_intervals
        ]
        pc = PatchCollection(rects, color=color)
        ax.add_collection(pc)


def show_psth_raster(
    data,
    before=0.5,
    after=2.0,
    group_inds=None,
    labels=None,
    ax=None,
    show_legend=True,
    align_line_color=(0.7, 0.7, 0.7),
    progress_bar: FloatProgress = None,
) -> plt.Axes:
    """

    Parameters
    ----------
    data: array-like
    before: float, optional
    after: float, optional
    group_inds: array-like, optional
    labels: array-like, optional
    ax: plt.Axes, optional
    show_legend: bool, optional
    align_line_color: array-like, optional
        [R, G, B] (0-1)
        Default = [0.7, 0.7, 0.7]
    progress_bar: FloatProgress, optional

    Returns
    -------
    plt.Axes

    """
    if not len(data):
        return ax
    ax = plot_grouped_events(
        data,
        [-before, after],
        group_inds,
        color_wheel,
        ax,
        labels,
        show_legend=show_legend,
        progress_bar=progress_bar,
    )
    ax.set_ylabel("trials")
    ax.axvline(color=align_line_color)
    return ax


def raster_grid(
    units: pynwb.misc.Units,
    time_intervals: pynwb.epoch.TimeIntervals,
    index,
    before,
    after,
    rows_label=None,
    cols_label=None,
    trials_select=None,
    align_by="start_time",
) -> plt.Figure:
    """

    Parameters
    ----------
    units: pynwb.misc.Units
    time_intervals: pynwb.epoch.TimeIntervals
    index: int
    before: float
    after: float
    rows_label: str, optional
    cols_label: str, optional
    trials_select: np.array(dtype=bool), optional
    align_by: str, optional

    Returns
    -------
    plt.Figure

    """
    if time_intervals is None:
        raise ValueError("trials must exist (trials cannot be None)")

    if trials_select is None:
        trials_select = np.ones((len(time_intervals),)).astype("bool")

    if rows_label is not None:
        row_vals = np.array(time_intervals[rows_label][:])
        urow_vals = np.unique(row_vals[trials_select])
        if urow_vals.dtype == np.float64:
            urow_vals = urow_vals[~np.isnan(urow_vals)]

    else:
        urow_vals = [None]
    nrows = len(urow_vals)

    if cols_label is not None:
        col_vals = np.array(time_intervals[cols_label][:])
        ucol_vals = np.unique(col_vals[trials_select])
        if ucol_vals.dtype == np.float64:
            ucol_vals = ucol_vals[~np.isnan(ucol_vals)]

    else:
        ucol_vals = [None]
    ncols = len(ucol_vals)

    fig, axs = plt.subplots(
        nrows, ncols, sharex=True, sharey=True, squeeze=False, figsize=(10, 10)
    )
    big_ax = create_big_ax(fig)
    for i, row in enumerate(urow_vals):
        for j, col in enumerate(ucol_vals):
            ax = axs[i, j]
            ax_trials_select = trials_select.copy()
            if row is not None:
                ax_trials_select &= row_vals == row
            if col is not None:
                ax_trials_select &= col_vals == col
            ax_trials_select = np.where(ax_trials_select)[0]
            if len(ax_trials_select):
                data = align_by_time_intervals(
                    units,
                    index,
                    time_intervals,
                    align_by,
                    align_by,
                    before,
                    after,
                    ax_trials_select,
                )
                show_psth_raster(data, before, after, ax=ax)
                ax.set_xlabel("")
                ax.set_ylabel("")
                if ax.get_subplotspec().is_first_col():
                    ax.set_ylabel(row)
                if ax.get_subplotspec().is_last_row():
                    ax.set_xlabel(col)

    big_ax.set_xlabel(cols_label, labelpad=50)
    big_ax.set_ylabel(rows_label, labelpad=60)

    return fig


def plot_grouped_events_plotly(
    data,
    window=None,
    group_inds=None,
    colors=color_wheel,
    labels=None,
    show_legend=True,
    unobserved_intervals_list=None,
    progress_bar=None,
    fig=None,
    **kwargs,
):
    data = np.array(data, dtype=object)

    if fig is None:
        fig = go.FigureWidget()
    if group_inds is not None:
        ugroup_inds = np.unique(group_inds)
        offset = 0
        for i in np.arange(len(ugroup_inds)):
            ui = ugroup_inds[i]
            color = colors[ugroup_inds[i] % len(colors)]
            this_data = data[group_inds == ui]
            event_group(
                this_data,
                offset=offset,
                label=labels[ui],
                color=color,
                fig=fig,
                **kwargs,
            )
            offset += len(this_data)

    else:
        event_group(data, fig=fig, **kwargs)

    fig.update_layout(xaxis_title="time (s)")

    return fig


class RasterWidgetPlotly(widgets.HBox):
    def __init__(
        self,
        units: Units,
        foreign_time_window_controller: StartAndDurationController = None,
        foreign_group_and_sort_controller: GroupAndSortController = None,
        group_by=None,
        fig: go.FigureWidget = None,
    ):
        super().__init__()

        self.units = units

        if foreign_time_window_controller is None:
            self.tmin = get_min_spike_time(units)
            self.tmax = get_max_spike_time(units)
            self.time_window_controller = StartAndDurationController(
                tmin=self.tmin, tmax=self.tmax
            )
        else:
            self.time_window_controller = foreign_time_window_controller

        if foreign_group_and_sort_controller:
            self.gas = foreign_group_and_sort_controller
        else:
            self.gas = GroupAndSortController(
                dynamic_table=self.units, group_by=group_by
            )

        self.show_legend_cb = widgets.Checkbox(value=True, description="show legend")

        if fig is None:
            self.fig = go.FigureWidget()
            self.fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        else:
            self.fig = fig
        show_session_raster_plotly(
            self.units, self.fig, self.time_window_controller.value, **self.gas.value
        )

        # set children
        if foreign_time_window_controller:
            right_panel = widgets.VBox(
                children=[
                    self.progress_bar,
                    self.fig,
                ],
                layout=Layout(width="100%"),
            )
        else:
            right_panel = widgets.VBox(
                children=[self.time_window_controller, self.fig, self.show_legend_cb],
                layout=Layout(width="100%"),
            )

        if foreign_group_and_sort_controller:
            self.children = [right_panel]
        else:
            self.children = [self.gas, right_panel]

        self.layout = Layout(width="100%")

        self.time_window_controller.observe(self.update_fig, "value")
        self.gas.observe(self.update_fig, "value")
        self.show_legend_cb.observe(self.toggle_legend, "value")

    def toggle_legend(self, change):
        self.fig.update_layout(showlegend=self.show_legend_cb.value)

    def update_fig(self, change):
        time_window = self.time_window_controller.value
        gas_kwargs = self.gas.value
        with self.fig.batch_update():
            self.fig.data = None
            show_session_raster_plotly(self.units, self.fig, time_window, **gas_kwargs)


def show_session_raster_plotly(
    units: Units, fig, time_window=None, order=None, progress_bar=None, **kwargs
):
    """

    Parameters
    ----------
    units: pynwb.misc.Units
    time_window: [int, int]
    show_obs_intervals: bool
    order: array-like, optional
    group_inds: array-like, optional
    labels: array-like, optional
    show_legend: bool
        default = True
        Does not show legend if color_by is None or 'id'.
    progress_bar: FloatProgress, optional

    Returns
    -------
    go.FigureWidget

    """

    if time_window is None:
        time_window = [get_min_spike_time(units), get_max_spike_time(units)]

    if order is None:
        order = np.arange(len(units), dtype="int")

    if progress_bar:
        this_iter = ProgressBar(order, desc="reading spike data", leave=False)
        progress_bar = this_iter.container
    else:
        this_iter = order
    data = []
    for unit in this_iter:
        data.append(get_spike_times(units, unit, time_window))

    # if show_obs_intervals:
    #    unobserved_intervals_list = get_unobserved_intervals(units, time_window, order)
    # else:
    #    unobserved_intervals_list = None

    fig.update_yaxes(tickvals=[], ticktext=[])
    if len(order) <= 100:
        kwargs.update(marker="line-ns", line_width=2)
    else:
        kwargs.update(line_width=1)
    fig = plot_grouped_events_plotly(data=data, fig=fig, **kwargs)
    if len(order) <= 40:
        fig.update_yaxes(
            tickvals=np.arange(len(order)), ticktext=[str(i) for i in order]
        )

    fig.update_layout(
        title="units",
        xaxis_title="time (s)",
        legend=dict(x=1.0, y=0, traceorder="reversed"),
        xaxis=dict(range=time_window),
        yaxis=dict(range=[-0.5, len(order) + 0.5]),
    )

    return fig


class UnitsAndTrialsControllerWidget(widgets.VBox):
    InnerWidget = None

    def __init__(
        self,
        units: Units,
        trials: pynwb.epoch.TimeIntervals = None,
        unit_index=0,
        **kwargs
    ):
        """
        Creates a UnitsAndTrials controller that controls InnerWidget.

        Parameters
        ----------
        units: pynwb.misc.Units object
        trials: pynwb.epoch.TimeIntervals object
        unit_index: int
        """
        super().__init__()

        self.units = units
        self.kwargs = kwargs

        # Check if there is trials table and create controller
        if trials is None:
            self.trials = self.get_trials()
            if self.trials is None:
                self.children = [widgets.HTML("No trials present")]
                return
        else:
            self.trials = trials

        # Create variables choice dropdowns
        groups = self.get_groups(self.trials)
        self.rows_controller = widgets.Dropdown(
            options=[None] + list(groups), 
            description="rows",
            value=None
        )
        self.rows_controller.observe(self.rows_callback, names='value')

        self.cols_controller = widgets.Dropdown(
            options=[None] + list(groups), 
            description="cols",
            disabled=True,
        )

        # Unit controller
        unit_ids = self.units.id.data[:]
        n_units = len(unit_ids)
        self.unit_controller = widgets.Dropdown(
            options=[(str(unit_ids[x]), x) for x in range(n_units)],
            value=unit_index,
            description="unit",
        )

        # Trial event controller (align by) 
        self.trial_event_controller = make_trial_event_controller(self.trials)

        # Before / After controllers
        self.before_slider = widgets.FloatSlider(
            0.1, min=0, max=5.0, description="before (s)", continuous_update=False
        )
        self.after_slider = widgets.FloatSlider(
            1.0, min=0, max=5.0, description="after (s)", continuous_update=False
        )

        self.fixed = dict(
            units=self.units,
            time_intervals=self.trials,
        )

        self.controls = {
            "index": self.unit_controller,
            "after": self.after_slider,
            "before": self.before_slider,
            "align_by": self.trial_event_controller,
            "rows_label": self.rows_controller,
            "cols_label": self.cols_controller,
        }
        
        self.children = [
            self.unit_controller,
            self.rows_controller,
            self.cols_controller,
            self.trial_event_controller,
            self.before_slider,
            self.after_slider
        ]

    def get_trials(self):
        return self.units.get_ancestor("NWBFile").trials

    def get_groups(self, trials):
        return infer_categorical_columns(dynamic_table=trials)

    def rows_callback(self, change):
        """
        Gets triggered when self.rows_controller changes. Updates other dropdown options.
        """
        if change['new'] is None:
            self.cols_controller.disabled = True
            self.cols_controller.value = None
        else:
            self.cols_controller.disabled = False



class RasterGridWidget(widgets.VBox):
    def __init__(
        self,
        units: Units,
        trials: pynwb.epoch.TimeIntervals = None,
        unit_index=0,
        units_trials_controller=None,
    ):

        super().__init__()

        # Create Units and Trials controller
        if not units_trials_controller:
            units_trials_controller = UnitsAndTrialsControllerWidget(
                units=units,
                trials=trials,
                unit_index=unit_index
            )
            self.children = [units_trials_controller]

        self.fig = interactive_output(
            f=raster_grid, 
            controls=units_trials_controller.controls,
            fixed=units_trials_controller.fixed
        )

        self.children += tuple([self.fig])


class TuningCurveWidget(widgets.VBox):
    def __init__(
        self,
        units: Units,
        trials: pynwb.epoch.TimeIntervals = None,
        unit_index=0,
        units_trials_controller=None,
    ):

        super().__init__()
        self.children = []

        # Create Units and Trials controller
        if not units_trials_controller:
            units_trials_controller = UnitsAndTrialsControllerWidget(
                units=units,
                trials=trials,
                unit_index=unit_index
            )
            self.children = [units_trials_controller]

        self.fig = interactive_output(
            f=draw_tuning_curve, 
            controls=units_trials_controller.controls,
            fixed=units_trials_controller.fixed
        )

        self.children += tuple([self.fig])



class TuningCurveExtendedWidget(widgets.VBox):
    def __init__(
        self,
        units: Units,
        trials: pynwb.epoch.TimeIntervals = None,
        unit_index=0
    ):
        super().__init__()

        # Controller
        self.units_trials_controller = UnitsAndTrialsControllerWidget(
            units=units,
            trials=trials,
            unit_index=unit_index
        )

        # Tuning curve widget
        self.tuning_curve = TuningCurveWidget(
            units=units,
            trials=trials,
            unit_index=unit_index,
            units_trials_controller=self.units_trials_controller,
        )

        # Raster grid widget
        self.raster_grid = RasterGridWidget(
            units=units,
            trials=trials,
            unit_index=unit_index,
            units_trials_controller=self.units_trials_controller,
        )

        self.children = [
            self.units_trials_controller,
            self.tuning_curve,
            self.raster_grid
        ]


def draw_tuning_curve(
    units: pynwb.misc.Units,
    time_intervals: pynwb.epoch.TimeIntervals,
    index,
    before,
    after,
    rows_label=None,
    cols_label=None,
    align_by="start_time",
) -> plt.Figure:

    if rows_label is None:
        return widgets.HTML("Select at least one variable")

    # 1D histogram
    if cols_label is None:
        return draw_tuning_curve_1d(
            units,
            time_intervals,
            index,
            before,
            after,
            rows_label,
            align_by
        )
    
    return draw_tuning_curve_2d(
        units,
        time_intervals,
        index,
        before,
        after,
        rows_label,
        cols_label,
        align_by
    )


def draw_tuning_curve_1d(
    units: pynwb.misc.Units,
    time_intervals: pynwb.epoch.TimeIntervals,
    index,
    before,
    after,
    rows_label=None,
    align_by="start_time",
) -> plt.Figure:

    rows_data, var1_classes = extract_data_from_intervals(time_intervals[rows_label])

    avg_rates = []
    for v1 in var1_classes:
        indexes = [i for i, d in enumerate(rows_data) if d==v1]
        data = align_by_time_intervals(
            units=units,
            index=index,
            intervals=time_intervals,
            start_label=align_by,
            stop_label=align_by,
            before=before,
            after=after,
            rows_select=indexes
        )
        n_trials = len(data)
        n_spikes = len(np.hstack(data))
        duration = after + before
        avg_rates.append(n_spikes / (n_trials * duration)) 

    x = np.arange(len(var1_classes))  # the label locations
    si = sort_mixed_type_list(var1_classes)
    avg_rates_sorted = [avg_rates[i] for i in si]

    fig, ax = plt.subplots(figsize=(14, 7))
    width = 0.95  # the width of the bars
    rects1 = ax.bar(x, avg_rates_sorted, width)
    lines1 = ax.plot(x, avg_rates_sorted, '-o', color='k', lw=2)

    # Labels
    ax.set_ylabel('Avg rate')
    ax.set_xlabel(rows_label)
    ax.set_xticks(x)
    ax.set_xticklabels([var1_classes[i] for i in si], rotation=45)
    fig.tight_layout()


def draw_tuning_curve_2d(
    units: pynwb.misc.Units,
    time_intervals: pynwb.epoch.TimeIntervals,
    index,
    before,
    after,
    rows_label=None,
    cols_label=None,
    align_by="start_time",
) -> plt.Figure:

    rows_data, var1_classes = extract_data_from_intervals(time_intervals[rows_label])
    cols_data, var2_classes = extract_data_from_intervals(time_intervals[cols_label])

    avg_rates = np.zeros((len(var1_classes), len(var2_classes)))
    for i, v1 in enumerate(var1_classes):
        for j, v2 in enumerate(var2_classes):
            indexes1 = [ii for ii, d in enumerate(rows_data) if d==v1]
            indexes2 = [ii for ii, d in enumerate(cols_data) if d==v2]
            intersect = list(set(indexes1) & set(indexes2))
            if len(intersect) > 0:
                data = align_by_time_intervals(
                    units=units,
                    index=index,
                    intervals=time_intervals,
                    start_label=align_by,
                    stop_label=align_by,
                    before=before,
                    after=after,
                    rows_select=intersect
                )
                n_trials = len(data)
                n_spikes = len(np.hstack(data))
                duration = after + before
                avg_rates[i, j] = n_spikes / (n_trials * duration)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    pos = ax.imshow(avg_rates.T, origin='lower', cmap='Greys')
    cbar = fig.colorbar(pos, ax=ax)
    cbar.set_label('spikes / second')

    # Labels
    ax.set_xticks(np.arange(len(var1_classes)))
    ax.set_yticks(np.arange(len(var2_classes)))
    ax.set_xlabel(rows_label)
    ax.set_ylabel(cols_label)
    ax.set_xticklabels(var1_classes, rotation=45)
    ax.set_yticklabels(var2_classes, rotation=45)

    return fig


def sort_mixed_type_list(x):
    """Returns the indexes for a sorted list of mixed types"""
    x_num = list()
    x_num_i = list()
    x_oth = list()
    x_oth_i = list()
    l = len(x)
    for i, xx in enumerate(x):
        try:
            x_num.append(float(xx)) 
            x_num_i.append(i)
        except:
            x_oth.append(str(xx))
            x_oth_i.append(i)
    x_num_si = np.argsort(x_num)
    x_oth_si = np.argsort(x_oth)
    return [x_num_i[ii] for ii in x_num_si] + [x_oth_i[ii] for ii in x_oth_si]