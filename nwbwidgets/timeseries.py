import functools
from bisect import bisect
from abc import abstractmethod
from typing import Union, Optional

from natsort import natsorted
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ipywidgets import widgets, fixed, Layout
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS

from pynwb.epoch import TimeIntervals
from pynwb.base import TimeSeries, DynamicTable
from pynwb.ophys import RoiResponseSeries
from pynwb.ecephys import ElectricalSeries

from .controllers import (
    StartAndDurationController,
    GroupAndSortController,
    RangeController,
    ProgressBar,
)
from .utils.plotly import multi_trace
from .utils.timeseries import (
    get_timeseries_tt,
    get_timeseries_maxt,
    get_timeseries_mint,
    timeseries_time_to_ind,
    get_timeseries_in_units,
    bisect_timeseries_by_times,
)
from .utils.widgets import interactive_output, set_plotly_callbacks
from .controllers.misc import make_trial_event_controller


color_wheel = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def show_ts_fields(node):
    info = []
    for key in ("description", "unit", "resolution", "conversion"):
        info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    return widgets.VBox(info)


def show_timeseries_mpl(
    time_series: TimeSeries,
    time_window=None,
    ax=None,
    zero_start=False,
    xlabel=None,
    ylabel=None,
    title=None,
    figsize=None,
    **kwargs,
):
    """

    Parameters
    ----------
    time_series: TimeSeries
    time_window: [int int]
    ax: plt.Axes
    zero_start: bool
    xlabel: str
    ylabel: str
    title: str
    figsize: tuple, optional
    kwargs

    Returns
    -------

    """
    if time_window is not None:
        istart = timeseries_time_to_ind(time_series, time_window[0])
        istop = timeseries_time_to_ind(time_series, time_window[1])
    else:
        istart = 0
        istop = None

    return show_indexed_timeseries_mpl(
        time_series,
        istart=istart,
        istop=istop,
        ax=ax,
        zero_start=zero_start,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        figsize=figsize,
        **kwargs,
    )


def show_indexed_timeseries_mpl(
    node: TimeSeries,
    istart=0,
    istop=None,
    ax=None,
    zero_start=False,
    xlabel="time (s)",
    ylabel=None,
    title=None,
    figsize=None,
    neurodata_vis_spec=None,
    **kwargs,
):
    if ylabel is None and node.unit:
        ylabel = node.unit

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    tt = get_timeseries_tt(node, istart=istart, istop=istop)
    if zero_start:
        tt = tt - tt[0]
    data, unit = get_timeseries_in_units(node, istart=istart, istop=istop)

    ax.plot(tt, data, **kwargs)
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    ax.autoscale(enable=True, axis="x", tight=True)

    return ax


def show_indexed_timeseries_plotly(
    timeseries: TimeSeries,
    istart: int = 0,
    istop: int = None,
    time_window: list = None,
    trace_range: list = None,
    offsets=None,
    fig: go.FigureWidget = None,
    col=None,
    row=None,
    zero_start=False,
    scatter_kwargs: dict = None,
    figure_kwargs: dict = None,
):
    if istart != 0 or istop is not None:
        if time_window is not None:
            raise ValueError("input either time window or istart/stop but not both")
        if not (0 <= istart < timeseries.data.shape[0] and (istop is None or 0 < istop <= timeseries.data.shape[0])):
            raise ValueError("enter correct istart/stop values")
        t_istart = istart
        t_istop = istop
    elif time_window is not None:
        t_istart = timeseries_time_to_ind(timeseries, time_window[0])
        t_istop = timeseries_time_to_ind(timeseries, time_window[1])
    else:
        t_istart = istart
        t_istop = istop
    tt = get_timeseries_tt(timeseries, istart=t_istart, istop=t_istop)
    data, unit = get_timeseries_in_units(timeseries, istart=t_istart, istop=t_istop)
    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    if trace_range is not None:
        if not (0 <= trace_range[0] < data.shape[1] and 0 < trace_range[1] <= data.shape[1]):
            raise ValueError("enter correct trace range")
        trace_istart = trace_range[0]
        trace_istop = trace_range[1]
    else:
        trace_istart = 0
        trace_istop = data.shape[1]
    if offsets is None:
        offsets = np.zeros(trace_istop - trace_istart)
    if zero_start:
        tt = tt - tt[0]
    scatter_kwargs = dict() if scatter_kwargs is None else scatter_kwargs
    if fig is None:
        fig = go.FigureWidget(make_subplots(rows=1, cols=1))
    row = 1 if row is None else row
    col = 1 if col is None else col
    for i, trace_id in enumerate(range(trace_istart, trace_istop)):
        fig.add_trace(
            go.Scattergl(x=tt, y=data[:, trace_id] + offsets[i], mode="lines", **scatter_kwargs),
            row=row,
            col=col,
        )
    input_figure_kwargs = dict(
        xaxis=dict(title_text="time (s)", range=[tt[0], tt[-1]]),
        yaxis=dict(title_text=unit if unit is not None else None),
        title=timeseries.name,
    )
    if figure_kwargs is None:
        figure_kwargs = dict()
    input_figure_kwargs.update(figure_kwargs)
    fig.update_xaxes(input_figure_kwargs.pop("xaxis"), row=row, col=col)
    fig.update_yaxes(input_figure_kwargs.pop("yaxis"), row=row, col=col)
    fig.update_layout(**input_figure_kwargs)
    return fig


def plot_traces(
    timeseries: TimeSeries,
    time_window=None,
    trace_window=None,
    title: str = None,
    ylabel: str = "traces",
    **kwargs,
):
    """

    Parameters
    ----------
    timeseries: TimeSeries
    time_window: [float, float], optional
        Start time and end time in seconds.
    trace_window: [int int], optional
        Index range of traces to view
    title: str, optional
    ylabel: str, optional

    Returns
    -------

    """

    if time_window is None:
        t_ind_start = 0
        t_ind_stop = None
    else:
        t_ind_start = timeseries_time_to_ind(timeseries, time_window[0])
        t_ind_stop = timeseries_time_to_ind(timeseries, time_window[1])

    if trace_window is None:
        trace_window = [0, timeseries.data.shape[1]]
    tt = get_timeseries_tt(timeseries, t_ind_start, t_ind_stop)
    if timeseries.data.shape[1] == len(tt):  # fix of orientation is incorrect
        mini_data = timeseries.data[trace_window[0] : trace_window[1], t_ind_start:t_ind_stop].T
    else:
        mini_data = timeseries.data[t_ind_start:t_ind_stop, trace_window[0] : trace_window[1]]

    gap = np.median(np.nanstd(mini_data, axis=0)) * 20
    offsets = np.arange(trace_window[1] - trace_window[0]) * gap

    fig, ax = plt.subplots()
    ax.figure.set_size_inches(12, 6)
    ax.plot(tt, mini_data + offsets, **kwargs)
    ax.set_xlabel("time (s)")
    if np.isfinite(gap):
        ax.set_ylim(-gap, offsets[-1] + gap)
        ax.set_xlim(tt[0], tt[-1])
        ax.set_yticks(offsets)
        ax.set_yticklabels(np.arange(trace_window[0], trace_window[1]))

    if title is not None:
        ax.set_title(title)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return fig


def show_timeseries(node, **kwargs):
    if len(node.data.shape) == 1:
        return SingleTracePlotlyWidget(node, **kwargs)
    elif len(node.data.shape) == 2:
        return BaseGroupedTraceWidget(node, **kwargs)
    else:
        raise ValueError(
            "Visualization for TimeSeries that has data with shape {} not implemented".format(node.data.shape)
        )


class AbstractTraceWidget(widgets.VBox):
    def __init__(
        self,
        timeseries: TimeSeries,
        foreign_time_window_controller: StartAndDurationController = None,
        **kwargs,
    ):
        super().__init__()
        self.timeseries = timeseries
        self.foreign_time_window_controller = foreign_time_window_controller
        self.controls = {}
        self.out_fig = None

        if foreign_time_window_controller is None:
            tmin = get_timeseries_mint(timeseries)
            tmax = get_timeseries_maxt(timeseries)
            self.time_window_controller = StartAndDurationController(tmax, tmin)
        else:
            self.time_window_controller = foreign_time_window_controller

        self.set_controls(**kwargs)
        self.set_out_fig()
        self.set_children()

    @abstractmethod
    def set_out_fig(self):
        return

    def set_controls(self, **kwargs):
        self.controls.update(timeseries=fixed(self.timeseries), time_window=self.time_window_controller)
        self.controls.update({key: widgets.fixed(val) for key, val in kwargs.items()})

    def set_children(self):
        if self.foreign_time_window_controller:
            self.children = [self.out_fig]
        else:
            self.children = [self.time_window_controller, self.out_fig]


class SingleTracePlotlyWidget(AbstractTraceWidget):
    def __init__(
        self,
        timeseries: TimeSeries,
        foreign_time_window_controller: StartAndDurationController = None,
        **kwargs,
    ):
        super().__init__(
            timeseries=timeseries,
            foreign_time_window_controller=foreign_time_window_controller,
            **kwargs,
        )

    def set_out_fig(self):
        timeseries = self.controls["timeseries"].value
        time_window = self.controls["time_window"].value
        self.out_fig = show_indexed_timeseries_plotly(timeseries=timeseries, time_window=time_window)

        def on_change(change):
            time_window = self.controls["time_window"].value
            istart = timeseries_time_to_ind(timeseries, time_window[0])
            istop = timeseries_time_to_ind(timeseries, time_window[1])
            yy, units = get_timeseries_in_units(timeseries, istart, istop)
            self.out_fig.data[0].x = get_timeseries_tt(timeseries, istart, istop)
            self.out_fig.data[0].y = list(yy)

            # Get data y-range, catching case with no data in current range (if so - no update)
            y_range = [min(yy), max(yy)] if yy.size != 0 else [None, None]
            self.out_fig.update_layout(
                yaxis={"range": y_range, "autorange": False},
                xaxis={"range": time_window, "autorange": False},
            )

        self.controls["time_window"].observe(on_change)


class SeparateTracesPlotlyWidget(AbstractTraceWidget):
    def set_out_fig(self):

        timeseries = self.controls["timeseries"].value

        time_window = self.controls["time_window"].value

        if len(timeseries.data.shape) > 1:
            color = DEFAULT_PLOTLY_COLORS
            no_rows = timeseries.data.shape[1]
            self.out_fig = go.FigureWidget(make_subplots(rows=no_rows, cols=1))

            for i, xyz in enumerate(("x", "y", "z")[:no_rows]):
                self.out_fig = show_indexed_timeseries_plotly(
                    timeseries=timeseries,
                    time_window=time_window,
                    trace_range=[i, i + 1],
                    fig=self.out_fig,
                    col=1,
                    row=i + 1,
                    scatter_kwargs=dict(marker_color=color[i % len(color)], name=xyz),
                    figure_kwargs=dict(yaxis=dict(title_text=xyz)),
                )
        else:
            self.out_fig = show_indexed_timeseries_plotly(timeseries=timeseries, time_window=time_window)

        def on_change(change):
            time_window = self.controls["time_window"].value
            istart = timeseries_time_to_ind(timeseries, time_window[0])
            istop = timeseries_time_to_ind(timeseries, time_window[1])

            tt = get_timeseries_tt(timeseries, istart, istop)
            yy, units = get_timeseries_in_units(timeseries, istart, istop)

            with self.out_fig.batch_update():
                if len(yy.shape) == 1:
                    self.out_fig.data[0].x = tt
                    self.out_fig.data[0].y = yy
                    self.out_fig.update_yaxes(range=[min(yy), max(yy)], row=1, col=1)
                    self.out_fig.update_xaxes(range=[min(tt), max(tt)], row=1, col=1)
                else:
                    for i, dd in enumerate(yy.T):
                        self.out_fig.data[i].x = tt
                        self.out_fig.data[i].y = dd
                        self.out_fig.update_yaxes(
                            range=[min(dd), max(dd)] if dd.size != 0 else [None, None],
                            row=i + 1,
                            col=1,
                        )
                        self.out_fig.update_xaxes(range=time_window, row=i + 1, col=1)

        self.controls["time_window"].observe(on_change)


def _prep_timeseries(time_series: TimeSeries, time_window=None, order=None):
    """Pull dataset region from entire dataset. Return tt and offests used for plotting

    Parameters
    ----------
    time_series: TimeSeries
    time_window
    order

    Returns
    -------

    """
    if time_window is None:
        t_ind_start = 0
        t_ind_stop = None
    else:
        t_ind_start = timeseries_time_to_ind(time_series, time_window[0])
        t_ind_stop = timeseries_time_to_ind(time_series, time_window[1])

    tt = get_timeseries_tt(time_series, t_ind_start, t_ind_stop)
    unique_sorted_order, inverse_sort = np.unique(order, return_inverse=True)

    if len(time_series.data.shape) > 1:
        mini_data = time_series.data[t_ind_start:t_ind_stop, unique_sorted_order][:, inverse_sort]
        if np.all(np.isnan(mini_data)):
            return None, tt, None
        gap = np.median(np.nanstd(mini_data, axis=0)) * 20
        offsets = np.arange(len(order)) * gap
        mini_data = mini_data + offsets
    else:
        mini_data = time_series.data[t_ind_start:t_ind_stop]
        offsets = [0]

    return mini_data, tt, offsets


def plot_grouped_traces(
    time_series: TimeSeries,
    time_window=None,
    order=None,
    ax=None,
    figsize=(8, 7),
    group_inds=None,
    labels=None,
    colors=color_wheel,
    show_legend=True,
    dynamic_table_region_name=None,
    window=None,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if order is None:
        if len(time_series.data.shape) > 1:
            order = np.arange(time_series.data.shape[1])
        else:
            order = [0]

    if group_inds is not None:
        row_ids = getattr(time_series, dynamic_table_region_name).data[:]
        channel_inds = [np.argmax(row_ids == x) for x in order]
    elif window is not None:
        order = order[window[0] : window[1]]
        channel_inds = order
    else:
        channel_inds = order

    if len(channel_inds):
        mini_data, tt, offsets = _prep_timeseries(time_series, time_window, channel_inds)
    else:
        mini_data = None
        tt = time_window

    if mini_data is None:
        ax.plot(tt, np.ones_like(tt) * np.nan, color="k")
        return

    if group_inds is not None:
        ugroup_inds = np.unique(group_inds)
        handles = []

        for i, ui in enumerate(ugroup_inds):
            color = colors[ugroup_inds[i] % len(colors)]
            lines_handle = ax.plot(tt, mini_data[:, group_inds == ui], color=color)
            handles.append(lines_handle[0])

        if show_legend:
            ax.legend(
                handles=handles[::-1],
                labels=list(labels[ugroup_inds][::-1]),
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
            )
    else:
        ax.plot(tt, mini_data, color="k")

    ax.set_xlim((tt[0], tt[-1]))
    ax.set_xlabel("time (s)")

    if len(offsets) > 1:
        ax.set_ylim(
            offsets[0] - (offsets[1] - offsets[0]) / 2,
            offsets[-1] + (offsets[-1] - offsets[-2]) / 2,
        )
    if len(order) <= 30:
        ax.set_yticks(offsets)
        ax.set_yticklabels(order)
    else:
        ax.set_yticks([])


def plot_grouped_traces_plotly(
    time_series: TimeSeries,
    time_window,
    order,
    group_inds=None,
    labels=None,
    colors=color_wheel,
    fig=None,
    **kwargs,
):
    mini_data, tt, offsets = _prep_timeseries(time_series, time_window, order)

    if fig is None:
        fig = go.FigureWidget()
    if group_inds is not None:
        ugroup_inds = np.unique(group_inds)
        for igroup, ui in enumerate(ugroup_inds[::-1]):
            color = colors[ugroup_inds[::-1][igroup] % len(colors)]
            group_data = mini_data[:, group_inds == ui].T
            multi_trace(tt, group_data, color, labels[ui], fig=fig)
    else:
        multi_trace(tt, mini_data.T, "black", fig=fig)
    fig.update_layout(title=time_series.name, xaxis_title="time (s)")
    fig.update_yaxes(tickvals=list(offsets), ticktext=[str(i) for i in order])

    return fig


class BaseGroupedTraceWidget(widgets.HBox):
    def __init__(
        self,
        time_series: TimeSeries,
        dynamic_table_region_name=None,
        foreign_time_window_controller: StartAndDurationController = None,
        foreign_group_and_sort_controller: GroupAndSortController = None,
        mpl_plotter=plot_grouped_traces,
        **kwargs,
    ):
        """

        Parameters
        ----------
        time_series: TimeSeries
        dynamic_table_region_name: str, optional
        foreign_time_window_controller: StartAndDurationController, optional
        foreign_group_and_sort_controller: GroupAndSortController, optional
        mpl_plotter: function
            Choose function to use when creating figures
        kwargs
        """

        if dynamic_table_region_name is not None and foreign_group_and_sort_controller is not None:
            raise TypeError(
                "You cannot supply both `dynamic_table_region_name` and `foreign_group_and_sort_controller`."
            )

        super().__init__()
        self.time_series = time_series

        if foreign_time_window_controller is not None:
            self.time_window_controller = foreign_time_window_controller
        else:
            self.tmin = get_timeseries_mint(time_series)
            self.tmax = get_timeseries_maxt(time_series)
            self.time_window_controller = StartAndDurationController(tmin=self.tmin, tmax=self.tmax)

        self.controls = dict(
            time_series=widgets.fixed(self.time_series),
            time_window=self.time_window_controller,
            dynamic_table_region_name=widgets.fixed(dynamic_table_region_name),
        )
        if foreign_group_and_sort_controller is None:
            if dynamic_table_region_name is not None:
                dynamic_table_region = getattr(time_series, dynamic_table_region_name)
                table = dynamic_table_region.table
                referenced_rows = np.array(dynamic_table_region.data)
                self.gas = GroupAndSortController(
                    dynamic_table=table,
                    keep_rows=referenced_rows,
                )
                self.controls.update(gas=self.gas)
            else:
                self.gas = None
                range_controller_max = min(30, self.time_series.data.shape[1])
                self.range_controller = RangeController(
                    0,
                    self.time_series.data.shape[1],
                    start_value=(0, range_controller_max),
                    dtype="int",
                    description="traces",
                    orientation="vertical",
                )
                self.controls.update(window=self.range_controller)
        else:
            self.gas = foreign_group_and_sort_controller
            self.controls.update(gas=self.gas)

        # Sets up interactive output controller
        out_fig = interactive_output(mpl_plotter, self.controls)

        if foreign_time_window_controller:
            right_panel = out_fig
        else:
            right_panel = widgets.VBox(
                children=[
                    self.time_window_controller,
                    out_fig,
                ],
                layout=widgets.Layout(width="100%"),
            )

        if foreign_group_and_sort_controller or self.gas is None:
            if self.range_controller is None:
                self.children = [right_panel]
            else:
                self.children = [self.range_controller, right_panel]
        else:

            self.children = [self.gas, right_panel]

        # self.layout = widgets.Layout(width="100%")


class MultiTimeSeriesWidget(widgets.VBox):
    def __init__(self, time_series_list, widget_class_list, constrain_time_range=False):
        """

        Parameters
        ----------
        time_series_list: list of TimeSeries
        widget_class_list: list of classes, optional
        constrain_time_range: bool, optional
            Default is False
        """
        super().__init__()
        if constrain_time_range:
            self.tmin = max(get_timeseries_mint(time_series) for time_series in time_series_list)
            self.tmax = min(get_timeseries_maxt(time_series) for time_series in time_series_list)
        else:
            self.tmin = min(get_timeseries_mint(time_series) for time_series in time_series_list)
            self.tmax = max(get_timeseries_maxt(time_series) for time_series in time_series_list)
        self.time_window_controller = StartAndDurationController(tmin=self.tmin, tmax=self.tmax)

        ts_widgets = [
            widget_class(time_series, foreign_time_window_controller=self.time_window_controller)
            for widget_class, time_series in zip(widget_class_list, time_series_list)
        ]
        self.children = [self.time_window_controller] + ts_widgets


class AlignMultiTraceTimeSeriesByTrialsAbstract(widgets.VBox):
    def __init__(
        self,
        time_series: TimeSeries,
        trials: TimeIntervals = None,
        trace_index=0,
        trace_controller=None,
        trace_controller_kwargs=None,
        sem=True,
    ):

        self.time_series = time_series
        self.time_series_data = time_series.data[()]
        self.time_series_timestamps = None
        if time_series.rate is None:
            self.time_series_timestamps = time_series.timestamps[()]
        super().__init__()

        if trials is None:
            self.trials = self.get_trials()
            if self.trials is None:
                self.children = [widgets.HTML("No trials present")]
                return
        else:
            self.trials = trials

        if trace_controller is None:
            ntraces = self.time_series.data.shape[1]
            input_trace_controller_kwargs = dict(
                options=[x for x in range(ntraces)],
                value=trace_index,
                description="trace",
                layout=Layout(width="200px"),
            )
            if trace_controller_kwargs is not None:
                input_trace_controller_kwargs.update(trace_controller_kwargs)
            self.trace_controller = widgets.Dropdown(**input_trace_controller_kwargs)
        else:
            self.trace_controller = trace_controller

        self.trial_event_controller = make_trial_event_controller(self.trials, layout=Layout(width="200px"))

        self.before_ft = widgets.FloatText(0.5, min=0, description="before (s)", layout=Layout(width="200px"))
        self.after_ft = widgets.FloatText(2.0, min=0, description="after (s)", layout=Layout(width="200px"))

        self.gas = self.make_group_and_sort(window=False, control_order=False)

        self.align_to_zero_cb = widgets.Checkbox(description="align to zero")

        self.controls = dict(
            index=self.trace_controller,
            after=self.after_ft,
            before=self.before_ft,
            start_label=self.trial_event_controller,
            gas=self.gas,
            align_to_zero=self.align_to_zero_cb,
        )
        vbox_cols = [
            [self.gas, self.align_to_zero_cb],
            [
                self.trace_controller,
                self.trial_event_controller,
                self.before_ft,
                self.after_ft,
            ],
        ]
        if sem:
            self.sem_cb = widgets.Checkbox(description="show SEM")
            self.controls.update(sem=self.sem_cb)
            vbox_cols[0].append(self.sem_cb)
        out_fig = set_plotly_callbacks(self.update, self.controls)

        self.children = [widgets.HBox([widgets.VBox(i) for i in vbox_cols]), out_fig]

    def get_trials(self):
        return self.time_series.get_ancestor("NWBFile").trials

    def make_group_and_sort(self, window=None, control_order=False, control_limit=False):
        return GroupAndSortController(
            self.trials,
            window=window,
            control_order=control_order,
            control_limit=control_limit,
        )

    def plot_group(self, group_inds, data_trialized, time_trialized, fig, order):
        for group in np.unique(group_inds):
            line_color = color_wheel[group % len(color_wheel)]
            pb = ProgressBar(
                np.where(group_inds == group)[0],
                desc=f"plotting {group} data",
                leave=False,
            )
            group_data = []
            group_ts = []
            for i, trim_trial_no in enumerate(pb):
                trial_idx = order[trim_trial_no]
                group_data.append(data_trialized[trial_idx])
                group_ts.append(time_trialized[trial_idx])
            fig = multi_trace(
                group_ts,
                group_data,
                fig=fig,
                color=line_color,
                label=str(group),
                insert_nans=True,
            )
        tt_flat = np.concatenate(time_trialized)
        fig.update_layout(
            xaxis_title="time (s)",
            yaxis_title=self.time_series.name,
            xaxis_range=(np.min(tt_flat), np.max(tt_flat)),
        )
        return fig


class AlignMultiTraceTimeSeriesByTrialsConstant(AlignMultiTraceTimeSeriesByTrialsAbstract):
    def __init__(
        self,
        time_series: TimeSeries,
        trials: TimeIntervals = None,
        trace_index=0,
        trace_controller=None,
        trace_controller_kwargs=None,
    ):

        self.time_series = time_series

        super().__init__(
            time_series=time_series,
            trials=trials,
            trace_index=trace_index,
            trace_controller=trace_controller,
            trace_controller_kwargs=trace_controller_kwargs,
            sem=True,
        )

    @functools.lru_cache()
    def align_data(self, start_label, before, after, index=None):
        starts = np.array(self.trials[start_label][:]) - before
        out_data_aligned = []
        out_ts_aligned = []
        for start in starts:
            idx_start = int((start - self.time_series.starting_time) * self.time_series.rate)
            idx_stop = int(idx_start + (before + after) * self.time_series.rate)
            out_ts_aligned.append(np.linspace(-before, after, num=idx_stop - idx_start))
            if len(self.time_series_data.shape) > 1 and index is not None:
                out_data_aligned.append(self.time_series_data[idx_start:idx_stop, index])
            else:
                out_data_aligned.append(self.time_series_data[idx_start:idx_stop])
        return out_data_aligned, out_ts_aligned

    def update(
        self,
        index: int,
        start_label: str = "start_time",
        before: float = 0.0,
        after: float = 1.0,
        order=None,
        group_inds=None,
        labels=None,
        align_to_zero=False,
        sem=False,
        fig: go.FigureWidget = None,
    ):
        data, time_ts_aligned = self.align_data(start_label, before, after, index)
        if group_inds is None:
            group_inds = np.zeros(len(self.trials), dtype=int)
        if align_to_zero:
            for trial_no in order:
                data_zero_id = bisect(time_ts_aligned[trial_no], 0)
                data[trial_no] -= data[trial_no][data_zero_id]
        fig = go.FigureWidget() if fig is None else fig
        fig.data = []
        fig.layout = {}
        if sem:
            group_stats = []
            for group in np.unique(group_inds):
                this_mean = np.nanmean(data[group_inds == group, :], axis=0)
                err = scipy.stats.sem(data[group_inds == group, :], axis=0, nan_policy="omit")
                group_stats.append(
                    dict(
                        mean=this_mean,
                        lower=this_mean - 2 * err,
                        upper=this_mean + 2 * err,
                        group=group,
                    )
                )

            for stats in group_stats:
                plot_kwargs = dict()
                color = color_wheel[stats["group"]]
                if labels is not None:
                    plot_kwargs.update(text=labels[stats["group"]])
                fig.add_scattergl(x=time_ts_aligned[0], y=stats["lower"], line_color=color)
                fig.add_scattergl(
                    x=time_ts_aligned[0],
                    y=stats["upper"],
                    line_color=color,
                    fill="tonexty",
                    opacity=0.2,
                )
                fig.add_scattergl(
                    x=time_ts_aligned[0],
                    y=stats["mean"],
                    line_color=color,
                    **plot_kwargs,
                )

        else:
            fig = self.plot_group(group_inds, data, time_ts_aligned, fig, order)
        return fig


class AlignMultiTraceTimeSeriesByTrialsVariable(AlignMultiTraceTimeSeriesByTrialsAbstract):
    def __init__(
        self,
        time_series: TimeSeries,
        trials: TimeIntervals = None,
        trace_index=0,
        trace_controller=None,
        trace_controller_kwargs=None,
    ):

        self.time_series = time_series

        super().__init__(
            time_series=time_series,
            trials=trials,
            trace_index=trace_index,
            trace_controller=trace_controller,
            trace_controller_kwargs=trace_controller_kwargs,
            sem=False,
        )

    @functools.lru_cache()
    def align_data(self, start_label, before, after, index=None):
        starts = np.array(self.trials[start_label][:]) - before
        out_data_aligned = []
        out_ts_aligned = []
        for start in starts:
            idx_start = bisect(self.time_series_timestamps, start - before)
            idx_stop = bisect(self.time_series_timestamps, start + after, lo=idx_start)
            out_ts_aligned.append(
                self.time_series_timestamps[idx_start:idx_stop] - self.time_series_timestamps[idx_start] - before
            )
            if len(self.time_series_data.shape) > 1 and index is not None:
                out_data_aligned.append(self.time_series_data[idx_start:idx_stop, index])
            else:
                out_data_aligned.append(self.time_series_data[idx_start:idx_stop])
        return out_data_aligned, out_ts_aligned

    def update(
        self,
        index: int,
        start_label: str = "start_time",
        before: float = 0.0,
        after: float = 1.0,
        order=None,
        group_inds=None,
        labels=None,
        align_to_zero=False,
        fig: go.FigureWidget = None,
    ):

        data, time_ts_aligned = self.align_data(start_label, before, after, index)
        if group_inds is None:
            group_inds = np.zeros(len(self.trials), dtype=int)
        if align_to_zero:
            for trial_no in order:
                data_zero_id = bisect(time_ts_aligned[trial_no], 0)
                data[trial_no] -= data[trial_no][data_zero_id]
        fig = fig if fig is not None else go.FigureWidget()
        fig.data = []
        fig.layout = {}
        return self.plot_group(group_inds, data, time_ts_aligned, fig, order)


def trialize_time_series(
    time_series: TimeSeries,
    trials_table: Union[DynamicTable, pd.DataFrame],
    data_column: Optional[int] = 0,
    start_time_shift: float = 0,
    duration: float = 1,
    alignment_column: str = "start_time",
) -> pd.DataFrame:

    if isinstance(trials_table, DynamicTable):
        trials_table_df = trials_table.to_dataframe()
    else:
        trials_table_df = trials_table

    trials_table_df = trials_table_df.reset_index().rename(columns={"id": "trial"})

    # Map timestamps to the trial interval (start_time, stop_time)
    timestamps = get_timeseries_tt(node=time_series)

    # Get the left and right bounds of the trials to get data and time
    values_to_align_array = trials_table_df[alignment_column].to_numpy()
    trial_left_bound = values_to_align_array + start_time_shift
    trial_left_bound = trial_left_bound[
        trial_left_bound <= timestamps.max()
    ]  # Filter values outside of the time series to avoid extra computation
    trial_right_bound = trial_left_bound + duration

    trial_left_bound_index = np.searchsorted(timestamps, trial_left_bound)
    trial_right_bound_index = np.searchsorted(timestamps, trial_right_bound)
    interval_bounds = zip(trial_left_bound_index, trial_right_bound_index)

    data_list = []
    timestamps_list = []
    alignment_column_list = []
    for trial_index, (idx_start, idx_stop) in enumerate(interval_bounds):
        trial_data, unit = get_timeseries_in_units(
            time_series, istart=idx_start, istop=idx_stop, data_column=data_column
        )
        data_list.append(trial_data)
        timestamps_list.append(timestamps[idx_start:idx_stop])
        alignment_column_list.append([trials_table_df[alignment_column][trial_index]] * (idx_stop - idx_start))

    data_dict = {
        "data": np.concatenate(data_list),
        "timestamps": np.concatenate(timestamps_list),
        f"{alignment_column}": np.concatenate(alignment_column_list),
    }

    data_df = pd.DataFrame(data_dict)

    data_df_trialized = pd.merge(left=data_df, right=trials_table_df, on=alignment_column, how="left")
    data_df_trialized["centered_timestamps"] = data_df_trialized.timestamps - data_df_trialized[alignment_column]

    return data_df_trialized


def create_empty_figure(text="No data"):
    empty_figure = go.Figure()
    empty_figure.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": f"{text}",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 28},
            }
        ],
    )

    return empty_figure


def calculate_moving_average_over_trials(df, moving_average_window):

    df_sort = df.sort_values(by="centered_timestamps")
    df_sort["moving_average"] = df_sort["data"].rolling(moving_average_window).mean()

    return df_sort


def add_moving_average_traces(figure, df, facet_col, facet_row):

    if facet_col is None and facet_row is None:
        num_trials = df["trial"].unique().size
        moving_average_window = 2 * num_trials
        df_sort = calculate_moving_average_over_trials(df, moving_average_window)

        figure.add_scattergl(
            x=df_sort.centered_timestamps,
            y=df_sort.moving_average,
            name="moving_average",
            line=dict(color="black", width=4),
        )
    elif facet_col is not None and facet_row is None:

        col_faceting_values = natsorted(df[facet_col].dropna().unique())

        for col_index, col_face_value in enumerate(col_faceting_values):
            if isinstance(col_face_value, str):
                query_string = f"{facet_col}=='{col_face_value}'"
            else:
                query_string = f"{facet_col}=={col_face_value}"

            sub_df = df.query(query_string)
            # Calculate moving average
            num_trials = sub_df["trial"].unique().size
            moving_average_window = 2 * num_trials

            moving_average_df = calculate_moving_average_over_trials(sub_df, moving_average_window)

            figure.add_scattergl(
                x=moving_average_df.centered_timestamps,
                y=moving_average_df.moving_average,
                showlegend=False,
                line=dict(color="black", width=4),
                row=1,
                col=col_index + 1,
            )

    elif facet_col is None and facet_row is not None:

        row_faceting_values = natsorted(df[facet_row].dropna().unique())

        for row_index, row_face_value in enumerate(reversed(row_faceting_values)):
            if isinstance(row_face_value, str):
                query_string = f"{facet_row}=='{row_face_value}'"
            else:
                query_string = f"{facet_row}=={row_face_value}"

            sub_df = df.query(query_string)
            # Calculate moving average
            num_trials = sub_df["trial"].unique().size
            moving_average_window = 2 * num_trials

            moving_average_df = calculate_moving_average_over_trials(sub_df, moving_average_window)

            figure.add_scattergl(
                x=moving_average_df.centered_timestamps,
                y=moving_average_df.moving_average,
                showlegend=False,
                line=dict(color="black", width=4),
                row=row_index + 1,
                col=1,
            )
    else:
        col_faceting_values = natsorted(df[facet_col].dropna().unique())
        row_faceting_values = natsorted(df[facet_row].dropna().unique())

        for row_index, row_face_value in enumerate(reversed(row_faceting_values)):
            for col_index, col_face_value in enumerate(col_faceting_values):

                if isinstance(col_face_value, str):
                    col_query_string = f"{facet_col}=='{col_face_value}'"
                else:
                    col_query_string = f"{facet_col}=={col_face_value}"

                if isinstance(row_face_value, str):
                    row_query_string = f"{facet_row}=='{row_face_value}'"
                else:
                    row_query_string = f"{facet_row}=={row_face_value}"

                query_string = row_query_string + " and " + col_query_string

                sub_df = df.query(query_string)
                # Calculate moving average
                num_trials = sub_df["trial"].unique().size
                moving_average_window = 2 * num_trials

                moving_average_df = calculate_moving_average_over_trials(sub_df, moving_average_window)

                figure.add_scattergl(
                    x=moving_average_df.centered_timestamps,
                    y=moving_average_df.moving_average,
                    showlegend=False,
                    line=dict(color="black", width=4),
                    row=row_index + 1,
                    col=col_index + 1,
                )

    return figure


def build_faceting_figure(df, facet_col, facet_row, data_label="data", trial_label="trial"):

    if df.empty:
        empty_figure = create_empty_figure()
        return empty_figure

    # Get the category orders
    category_orders = dict()
    if facet_row:
        row_faceting_values = natsorted(df[facet_row].dropna().unique())
        category_orders.update({facet_row: row_faceting_values})
    if facet_col:
        col_faceting_values = natsorted(df[facet_col].dropna().unique())
        category_orders.update({facet_col: col_faceting_values})

    # Drop NA values
    faceting_values = [facet_row, facet_col]
    faceting_values = [value for value in faceting_values if value is not None]
    if faceting_values:
        df = df.dropna(subset=faceting_values)

    # Construct all the traces grouped by trial
    figure = px.line(
        df,
        x="centered_timestamps",
        y=data_label,
        color=trial_label,
        facet_col=facet_col,
        facet_row=facet_row,
        category_orders=category_orders,
    )
    figure.update_traces(line_color="gray", line_width=1, showlegend=False)

    # Add moving average
    figure = add_moving_average_traces(figure, df, facet_col=facet_col, facet_row=facet_row)

    # Annotations
    figure.for_each_xaxis(lambda x: x.update(title=""))
    figure.for_each_yaxis(lambda y: y.update(title=""))
    figure.add_annotation(
        x=-0.165,
        y=0.5,
        textangle=270,
        text=f"{data_label}",
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    figure.add_annotation(
        x=0.5,
        y=-0.165,
        text="Centered timestamps (s)",
        xref="paper",
        yref="paper",
        showarrow=False,
    )

    if facet_row is not None:
        figure.add_annotation(
            x=1.075,
            y=0.5,
            textangle=90,
            text=f"{facet_row}",
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    if facet_col is not None:
        figure.add_annotation(
            x=0.5,
            y=1.15,
            textangle=0,
            text=f"{facet_col}",
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    annotation_list = figure.layout.annotations
    valid_annotations = (annotation for annotation in annotation_list if "=" in annotation.text)
    for annotation in valid_annotations:
        annotation.text = annotation.text.split("=")[1]

    return figure


class TrializedTimeSeriesController(widgets.VBox):
    def __init__(
        self, time_series: TimeSeries, trials_table: Optional[DynamicTable] = None, column_selection_text="Data col"
    ):
        super().__init__()

        self.time_series = time_series
        self.trials_table = trials_table
        self.trials_table_df = self.trials_table.to_dataframe()

        # Labels to refer to data created by the widget. Should not collapse with the column names on the dynamic table
        self.timestamps_label = "timestamps"
        self.centered_timestamps_label = "centered_timestamps"

        self.available_columns = self.trials_table_df.columns

        invalid_columns = ["start_time", "stop_time"]
        invalid_columns += [
            self.timestamps_label,
            self.centered_timestamps_label,
        ]

        # Extract ragged columns (can't filter or facet with them)
        get_indexed_column_name = lambda col: "_".join(col.name.split("_")[:-1])
        ragged_columns = [get_indexed_column_name(col) for col in self.trials_table.columns if "index" in col.name]
        invalid_columns += ragged_columns

        self.invalid_columns = invalid_columns
        self.columns_for_filtering = natsorted(
            [column for column in self.trials_table_df.columns if column not in self.invalid_columns]
        )

        self.options_per_column = {
            column: natsorted(list(self.trials_table_df[column].dropna().unique()))
            for column in self.columns_for_filtering
        }

        # Define widgets
        self.columns_for_filtering.append(None)
        self.select_filter_columns = widgets.SelectMultiple(
            options=self.columns_for_filtering,
            description="Col to filter",
            value=[None],
        )

        self.filtering_active = False
        self.filter_menu = widgets.VBox()

        self.faceting_column_selection = widgets.Dropdown(
            value=None, options=self.columns_for_filtering, description="col faceting"
        )
        self.faceting_row_selection = widgets.Dropdown(
            value=None, options=self.columns_for_filtering, description="row faceting"
        )

        data_shape = self.time_series.data.shape
        dimension_options = list(range(data_shape[1]))
        self.data_column_selection = widgets.Dropdown(
            options=dimension_options,
            description=column_selection_text,
            description_tooltipw=f"{self.time_series.name} column to plot",
            value=0,
        )

        self.column_to_align_to_widget = make_trial_event_controller(trials=self.trials_table)
        self.start_time_shift_widget = widgets.FloatText(
            0.0,
            step=0.1,
            description="start (s)",
            layout=Layout(width="200px"),
            description_tooltip="Start time for calculation before or after (negative or positive) the reference point (aligned to)",
        )

        self.end_time_widget = widgets.FloatText(
            1.0,
            step=0.1,
            description="end (s)",
            layout=Layout(width="200px"),
            description_tooltip="End time for calculation before or after (negative or positive) the reference point (aligned to).",
        )

        self.match_x_axis_widget = widgets.Checkbox(
            value=True,
            description="Lock x-axis across facets",
            disabled=False,
            indent=False,
        )

        self.children = [
            self.data_column_selection,
            self.select_filter_columns,
            self.filter_menu,
            self.faceting_column_selection,
            self.faceting_row_selection,
            self.column_to_align_to_widget,
            self.start_time_shift_widget,
            self.end_time_widget,
            self.match_x_axis_widget,
        ]

        # Define observers to account for interactions between controls
        self.select_filter_columns.observe(self.update_filter_menu, names="value")
        self.select_filter_columns.observe(self.update_row_faceting, names="value")
        self.select_filter_columns.observe(self.update_column_faceting, names="value")

    def update_filter_menu(self, change):
        selected_columns = self.select_filter_columns.value
        if selected_columns != (None,):
            selection_boxes = [
                widgets.Dropdown(options=self.options_per_column[column], description=column)
                for column in selected_columns
            ]
            self.filter_menu.children = tuple(selection_boxes)
            self.filtering_active = True
            self.filter_menu.layout.visibility = "visible"

        else:
            self.filtering_active = False
            self.filter_menu.layout.visibility = "hidden"

    def update_row_faceting(self, change):
        non_selected_columns = list(set(self.columns_for_filtering).difference(self.select_filter_columns.value))

        self.faceting_row_selection.options = natsorted(list(non_selected_columns))
        self.faceting_row_selection.value = None

    def update_column_faceting(self, change):
        non_selected_columns = list(set(self.columns_for_filtering).difference(self.select_filter_columns.value))
        self.faceting_column_selection.options = natsorted(list(non_selected_columns))
        self.faceting_column_selection.value = None


class TrializedTimeSeries(widgets.HBox):
    def __init__(
        self, time_series: TimeSeries, trials_table: Optional[DynamicTable] = None, column_selection_text="Data col"
    ):
        super().__init__()

        self.time_series = time_series
        self.trials_table = trials_table
        if self.trials_table is None:
            self.trials_table = time_series.get_ancestor("NWBFile").trials

        if self.trials_table is None:
            self.children = [widgets.HTML("No trials present")]
            return

        self.trials_table_df = self.trials_table.to_dataframe()

        self.controller = TrializedTimeSeriesController(
            time_series=self.time_series, trials_table=self.trials_table, column_selection_text=column_selection_text
        )
        self.plot_button = widgets.Button(description="Plot selection!")

        empty_figure = create_empty_figure(text="Select configuration to plot")
        self.figure_widget = go.FigureWidget(empty_figure)

        # Register plotting button callback
        self.plot_button.on_click(self.update_plot_widget)

        # Create the structure
        self.control = widgets.VBox(
            [
                self.controller,
                self.plot_button,
            ]
        )
        self.children = [self.control, self.figure_widget]

    def query_expresion(self, children):
        if isinstance(children.value, str):
            return f" {children.description} == '{children.value}' "
        else:
            return f" {children.description} == {children.value} "

    def update_data_state(self):

        if self.controller.filtering_active:
            query_string = "and".join(
                [self.query_expresion(children) for children in self.controller.filter_menu.children]
            )
            df_query = self.trials_table.to_dataframe().query(query_string)
        else:
            df_query = self.trials_table.to_dataframe()

        self.trialized_data_df = trialize_time_series(
            time_series=self.time_series,
            trials_table=df_query,
            start_time_shift=self.controller.start_time_shift_widget.value,
            duration=(self.controller.end_time_widget.value - self.controller.start_time_shift_widget.value),
            data_column=self.controller.data_column_selection.value,
            alignment_column=self.controller.column_to_align_to_widget.value,
        )

    def update_plot_widget(self, button_instance):

        # Update data
        self.update_data_state()

        # Generate the plot
        facet_col = self.controller.faceting_column_selection.value
        facet_row = self.controller.faceting_row_selection.value
        figure = build_faceting_figure(df=self.trialized_data_df, facet_col=facet_col, facet_row=facet_row)

        # Update the widget
        with self.figure_widget.batch_update():
            self.figure_widget.update(layout_annotations=None)
            self.figure_widget.update(figure.to_dict(), overwrite=True)
            matches = "x" if self.controller.match_x_axis_widget.value else None
            self.figure_widget.update_xaxes(matches=matches)


def route_trialized_time_series(time_series: TimeSeries, neurodata_vis_spec=None, **kwargs):
    """Function to route different type of time series to the appropriate TrializedTimeSeries case

    Args:
        time_series (TimeSeries): A pynwb.TimeSeries object
        neurodata_vis_spec (_type_, optional): The general dictionary  that maps neuodatatypes in nwb to specific
        visualizations.
    """

    if isinstance(time_series, RoiResponseSeries):
        return TrializedTimeSeries(time_series=time_series, column_selection_text="ROI")
    elif isinstance(time_series, ElectricalSeries):
        return TrializedTimeSeries(time_series=time_series, column_selection_text="Electrode")
    else:
        return TrializedTimeSeries(time_series=time_series)
