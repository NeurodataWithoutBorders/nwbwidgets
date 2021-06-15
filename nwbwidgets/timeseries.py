from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect
import plotly.graph_objects as go
from ipywidgets import widgets, fixed, Layout
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS

from pynwb import TimeSeries
from pynwb.epoch import TimeIntervals
import scipy

from .controllers import (
    StartAndDurationController,
    GroupAndSortController,
    RangeController,
)
from .utils.plotly import multi_trace
from .utils.timeseries import (
    get_timeseries_tt,
    get_timeseries_maxt,
    get_timeseries_mint,
    timeseries_time_to_ind,
    get_timeseries_in_units,
)
from .utils.widgets import interactive_output, set_plotly_callbacks
from .controllers.misc import make_trial_event_controller
from .utils.timeseries import align_by_time_intervals, align_timestamps_by_trials

from .utils.dynamictable import infer_categorical_columns

color_wheel = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def show_ts_fields(node):
    info = []
    for key in ("description", "unit", "resolution", "conversion"):
        info.append(
            widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True)
        )
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
    **kwargs
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
    **kwargs
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
    istart=None,
    istop=None,
    fig: go.FigureWidget = None,
    col=None,
    row=None,
    zero_start=False,
    xlabel="time (s)",
    ylabel=None,
    title=None,
    neurodata_vis_spec=None,
    **kwargs
):
    if ylabel is None and timeseries.unit:
        ylabel = timeseries.unit

    tt = get_timeseries_tt(timeseries, istart=istart, istop=istop)
    if zero_start:
        tt = tt - tt[0]
    data, unit = get_timeseries_in_units(timeseries, istart=istart, istop=istop)

    trace_kwargs = dict()
    if col is not None or row is not None:
        trace_kwargs.update(row=row, col=col)
    fig.add_trace(x=tt, y=data, **trace_kwargs, **kwargs)
    layout_kwargs = dict(xaxis_title=xlabel)
    if ylabel is not None:
        layout_kwargs.update(yaxis_title=ylabel)
    if title is not None:
        layout_kwargs.update(title=title)

    fig.update_layout(**layout_kwargs)


def plot_traces(
    timeseries: TimeSeries,
    time_window=None,
    trace_window=None,
    title: str = None,
    ylabel: str = "traces",
    **kwargs
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
        mini_data = timeseries.data[
            trace_window[0] : trace_window[1], t_ind_start:t_ind_stop
        ].T
    else:
        mini_data = timeseries.data[
            t_ind_start:t_ind_stop, trace_window[0] : trace_window[1]
        ]

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
            "Visualization for TimeSeries that has data with shape {} not implemented".format(
                node.data.shape
            )
        )


class AbstractTraceWidget(widgets.VBox):
    def __init__(
        self,
        timeseries: TimeSeries,
        foreign_time_window_controller: StartAndDurationController = None,
        **kwargs
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

    def mpl_plotter(self, **kwargs):
        return

    @abstractmethod
    def set_children(self):
        return

    def set_controls(self, **kwargs):
        self.controls.update(
            timeseries=fixed(self.timeseries), time_window=self.time_window_controller
        )
        self.controls.update({key: widgets.fixed(val) for key, val in kwargs.items()})

    def set_out_fig(self):
        self.out_fig = widgets.interactive_output(self.mpl_plotter, self.controls)


class SingleTraceWidget(AbstractTraceWidget):
    mpl_plotter = show_timeseries

    def set_children(self):
        if self.foreign_time_window_controller:
            self.children = [self.out_fig]
        else:
            self.children = [self.time_window_controller, self.out_fig]


class SingleTracePlotlyWidget(SingleTraceWidget):
    def __init__(
        self,
        timeseries: TimeSeries,
        foreign_time_window_controller: StartAndDurationController = None,
        **kwargs
    ):
        super().__init__(
            timeseries=timeseries,
            foreign_time_window_controller=foreign_time_window_controller,
            **kwargs,
        )

    def set_out_fig(self):
        timeseries = self.controls["timeseries"].value
        time_window = self.controls["time_window"].value

        istart = timeseries_time_to_ind(timeseries, time_window[0])
        istop = timeseries_time_to_ind(timeseries, time_window[1])
        yy, units = get_timeseries_in_units(timeseries, istart, istop)

        self.out_fig = go.FigureWidget(
            data=go.Scatter(x=get_timeseries_tt(timeseries, istart, istop), y=list(yy))
        )

        self.out_fig.update_layout(
            title=timeseries.name,
            xaxis_title="time (s)",
            yaxis_title=units,
            yaxis={"range": [min(yy), max(yy)], "autorange": False},
            xaxis={
                "range": [min(self.out_fig.data[0].x), max(self.out_fig.data[0].x)],
                "autorange": False,
            },
        )

        def on_change(change):
            time_window = self.controls["time_window"].value
            istart = timeseries_time_to_ind(timeseries, time_window[0])
            istop = timeseries_time_to_ind(timeseries, time_window[1])
            yy, units = get_timeseries_in_units(timeseries, istart, istop)
            self.out_fig.data[0].x = get_timeseries_tt(timeseries, istart, istop)
            self.out_fig.data[0].y = list(yy)

            self.out_fig.update_layout(
                yaxis={"range": [min(yy), max(yy)], "autorange": False},
                xaxis={
                    "range": [min(self.out_fig.data[0].x), max(self.out_fig.data[0].x)],
                    "autorange": False,
                },
            )

        self.controls["time_window"].observe(on_change)


class SeparateTracesPlotlyWidget(SingleTraceWidget):
    def set_out_fig(self):

        timeseries = self.controls["timeseries"].value

        time_window = self.controls["time_window"].value

        istart = timeseries_time_to_ind(timeseries, time_window[0])
        istop = timeseries_time_to_ind(timeseries, time_window[1])

        data, units = get_timeseries_in_units(timeseries, istart, istop)

        tt = get_timeseries_tt(timeseries, istart, istop)

        if len(data.shape) > 1:
            color = DEFAULT_PLOTLY_COLORS[0]
            self.out_fig = go.FigureWidget(make_subplots(rows=data.shape[1], cols=1))

            for i, (yy, xyz) in enumerate(zip(data.T, ("x", "y", "z"))):
                self.out_fig.add_trace(
                    go.Scatter(x=tt, y=yy, marker_color=color), row=i + 1, col=1
                )
                if units:
                    yaxes_label = "{} ({})".format(xyz, units)
                else:
                    yaxes_label = xyz
                self.out_fig.update_yaxes(title_text=yaxes_label, row=i + 1, col=1)
            self.out_fig.update_xaxes(title_text="time (s)", row=i + 1, col=1)
        else:
            self.out_fig = go.FigureWidget()
            self.out_fig.add_trace(go.Scatter(x=tt, y=data))
            self.out_fig.update_xaxes(title_text="time (s)")

        self.out_fig.update_layout(showlegend=False, title=timeseries.name)

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
                else:
                    for i, dd in enumerate(yy.T):
                        self.out_fig.data[i].x = tt
                        self.out_fig.data[i].y = dd

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
        mini_data = time_series.data[t_ind_start:t_ind_stop, unique_sorted_order][
            :, inverse_sort
        ]
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
    **kwargs
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

    mini_data, tt, offsets = _prep_timeseries(time_series, time_window, channel_inds)

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
    **kwargs
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
        **kwargs
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

        if (
            dynamic_table_region_name is not None
            and foreign_group_and_sort_controller is not None
        ):
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
            self.time_window_controller = StartAndDurationController(
                tmin=self.tmin, tmax=self.tmax
            )

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
            self.tmin = max(
                get_timeseries_mint(time_series) for time_series in time_series_list
            )
            self.tmax = min(
                get_timeseries_maxt(time_series) for time_series in time_series_list
            )
        else:
            self.tmin = min(
                get_timeseries_mint(time_series) for time_series in time_series_list
            )
            self.tmax = max(
                get_timeseries_maxt(time_series) for time_series in time_series_list
            )
        self.time_window_controller = StartAndDurationController(
            tmin=self.tmin, tmax=self.tmax
        )

        widgets = [
            widget_class(
                time_series, foreign_time_window_controller=self.time_window_controller
            )
            for widget_class, time_series in zip(widget_class_list, time_series_list)
        ]
        self.children = [self.time_window_controller] + widgets


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

        self.trial_event_controller = make_trial_event_controller(
            self.trials, layout=Layout(width="200px")
        )

        self.before_ft = widgets.FloatText(
            0.5, min=0, description="before (s)", layout=Layout(width="200px")
        )
        self.after_ft = widgets.FloatText(
            2.0, min=0, description="after (s)", layout=Layout(width="200px")
        )

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

    def make_group_and_sort(
        self, window=None, control_order=False, control_limit=False
    ):
        return GroupAndSortController(
            self.trials,
            window=window,
            control_order=control_order,
            control_limit=control_limit,
        )


class AlignMultiTraceTimeSeriesByTrialsConstant(
    AlignMultiTraceTimeSeriesByTrialsAbstract
):
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
        fig:go.FigureWidget = None,
    ):
        data = align_by_time_intervals(
            self.time_series, self.trials, start_label, before, after, traces=index
        )
        rate = self.time_series.rate
        tt = np.arange(data.shape[1]) / rate - before

        if group_inds is None:
            group_inds = np.zeros(data.shape[0], dtype=np.int)
        group_stats = []

        data = data[order]

        if align_to_zero:
            data_zero_id = bisect(tt, 0)
            data = data - data[:, data_zero_id, np.newaxis]

        fig = go.FigureWidget() if fig is None else fig
        fig.data = []
        fig.layout = {}
        if sem:
            for group in np.unique(group_inds):
                this_mean = np.nanmean(data[group_inds == group, :], axis=0)
                err = scipy.stats.sem(
                    data[group_inds == group, :], axis=0, nan_policy="omit"
                )
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
                fig.add_scattergl(x=tt, y=stats["lower"], line_color=color)
                fig.add_scattergl(x=tt, y=stats["upper"], line_color=color, fill='tonexty', opacity=0.2)
                fig.add_scattergl(x=tt, y=stats["mean"], line_color=color, **plot_kwargs)

        else:
            for group in np.unique(group_inds):
                labels = labels[group] if labels is not None else str(group)
                fig=multi_trace(x=tt, y=data[group_inds == group, :],
                                  color=color_wheel[group%len(color_wheel)], label=labels, fig=fig)
        fig.update_layout(xaxis_title='time (s)',
                          yaxis_title=self.time_series.name,
                          xaxis_range=(np.min(tt), np.max(tt)))
        return fig

class AlignMultiTraceTimeSeriesByTrialsVariable(
    AlignMultiTraceTimeSeriesByTrialsAbstract
):
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
        fig:go.FigureWidget = None,
    ):
        data = align_by_time_intervals(
            self.time_series,
            self.trials,
            start_label,
            before,
            after,
            traces=index,
        )
        starts = np.array(self.trials[start_label][:]) - before
        time_ts_aligned = align_timestamps_by_trials(
            self.time_series, starts, before, after
        )

        if group_inds is None:
            group_inds = np.zeros(len(data), dtype=np.int)

        if align_to_zero:
            for trial_no in order:
                data_zero_id = bisect(time_ts_aligned[trial_no], 0)
                data[trial_no] -= data[trial_no][data_zero_id]
        fig = fig if fig is not None else go.FigureWidget()
        fig.data = []
        fig.layout = {}
        for group in np.unique(group_inds):
            for i,trim_trial_no in enumerate(np.where(group_inds==group)[0]):
                showlegend=True if i==0 else False
                plot_kwargs = dict()
                if labels is not None:
                    plot_kwargs.update(legendgroup=str(labels[group]),
                                       showlegend=showlegend,
                                       name=str(labels[group]))
                fig.add_scattergl(x=time_ts_aligned[order[trim_trial_no]],
                                  y=data[order[trim_trial_no]],
                                  line_color=color_wheel[group%len(color_wheel)],
                                  **plot_kwargs)
        tt_flat = np.concatenate(time_ts_aligned)
        fig.update_layout(xaxis_title='time (s)',
                          yaxis_title=self.time_series.name,
                          xaxis_range=(np.min(tt_flat), np.max(tt_flat)))
        return fig
