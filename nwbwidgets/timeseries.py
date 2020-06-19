import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets, fixed
from pynwb import TimeSeries
from .utils.timeseries import (get_timeseries_tt, get_timeseries_maxt, get_timeseries_mint,
                               timeseries_time_to_ind, get_timeseries_in_units)
from abc import abstractmethod
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .controllers import StartAndDurationController,  RangeController, GroupAndSortController
from .utils.widgets import interactive_output

color_wheel = plt.rcParams['axes.prop_cycle'].by_key()['color']


def show_ts_fields(node):
    info = []
    for key in ('description', 'unit', 'resolution', 'conversion'):
        info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    return widgets.VBox(info)


def show_timeseries_mpl(node: TimeSeries, neurodata_vis_spec=None, istart=0, istop=None, ax=None, zero_start=False,
                        xlabel=None, ylabel=None, title=None, **kwargs):
    if xlabel is None:
        xlabel = 'time (s)'

    if ylabel is None and node.unit:
        ylabel = node.unit

    if ax is None:
        fig, ax = plt.subplots()

    tt = get_timeseries_tt(node, istart=istart, istop=istop)
    if zero_start:
        tt = tt - tt[0]
    data, unit = get_timeseries_in_units(node, istart=istart, istop=istop)

    ax.plot(tt, data, **kwargs)
    ax.set_xlabel(xlabel)
    if node.unit:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    ax.autoscale(enable=True, axis='x', tight=True)

    return ax


def plot_traces(timeseries: TimeSeries, time_window=None, trace_window=None,
                title: str = None, ylabel: str = 'traces', **kwargs):
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
        mini_data = timeseries.data[trace_window[0]:trace_window[1], t_ind_start:t_ind_stop].T
    else:
        mini_data = timeseries.data[t_ind_start:t_ind_stop, trace_window[0]:trace_window[1]]

    gap = np.median(np.nanstd(mini_data, axis=0)) * 20
    offsets = np.arange(trace_window[1] - trace_window[0]) * gap

    fig, ax = plt.subplots()
    ax.figure.set_size_inches(12, 6)
    ax.plot(tt, mini_data + offsets, **kwargs)
    ax.set_xlabel('time (s)')
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
        return TracesWidget(node, **kwargs)
    else:
        raise ValueError('Visualization for TimeSeries that has data with shape {} not implemented'.
                         format(node.data.shape))


class AbstractTraceWidget(widgets.VBox):
    def __init__(self,
                 timeseries: TimeSeries,
                 foreign_time_window_controller: StartAndDurationController = None,
                 **kwargs):
        super().__init__()
        self.timeseries = timeseries
        self.foreign_time_window_controller = foreign_time_window_controller
        self.controls = {}
        self.out_fig = None

        if foreign_time_window_controller is None:
            tmin = get_timeseries_mint(timeseries)
            tmax = get_timeseries_maxt(timeseries)
            self.time_window_controller = StartAndDurationController(
                tmax, tmin, start_value=tmin, duration=min(5, tmax - tmin))
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
        self.controls.update(timeseries=fixed(self.timeseries), time_window=self.time_window_controller)
        self.controls.update({key: widgets.fixed(val) for key, val in kwargs.items()})

    def set_out_fig(self):
        self.out_fig = widgets.interactive_output(self.mpl_plotter, self.controls)


class SingleTraceWidget(AbstractTraceWidget):
    def __init__(self,
                 timeseries: TimeSeries,
                 foreign_time_window_controller: StartAndDurationController = None,
                 neurodata_vis_spec=None,
                 **kwargs):
        super().__init__(timeseries, foreign_time_window_controller, **kwargs)

    def mpl_plotter(self, timeseries, time_window, figsize=(12, 3), **kwargs):
        istart = timeseries_time_to_ind(timeseries, time_window[0])
        istop = timeseries_time_to_ind(timeseries, time_window[1])

        fig, ax = plt.subplots(figsize=figsize)

        return show_timeseries_mpl(timeseries, istart=istart, istop=istop, ax=ax)

    def set_children(self):
        if self.foreign_time_window_controller:
            self.children = [self.out_fig]
        else:
            self.children = [self.time_window_controller, self.out_fig]


class TracesWidget(AbstractTraceWidget):
    def __init__(self, timeseries: TimeSeries, neurodata_vis_spec: dict = None,
                 foreign_time_window_controller: StartAndDurationController = None,
                 foreign_traces_controller: RangeController = None, trace_starting_range=None,
                 **kwargs):

        if foreign_traces_controller is None:
            if trace_starting_range is None:
                trace_starting_range = (0, min(30, timeseries.data.shape[1]))
            self.trace_controller = RangeController(
                0, timeseries.data.shape[1], start_range=trace_starting_range, description='channels', dtype='int',
                orientation='vertical')
        else:
            self.trace_controller = foreign_traces_controller

        super().__init__(timeseries, foreign_time_window_controller, **kwargs)

    def set_children(self):

        self.children = (self.time_window_controller,
                         widgets.HBox(children=[
                             self.trace_controller,
                             self.out_fig
                         ])
                         )

    def set_controls(self, **kwargs):
        super().set_controls()
        self.controls.update(trace_window=self.trace_controller)

    def mpl_plotter(self, **kwargs):
        return plot_traces(**kwargs)


class SingleTracePlotlyWidget(SingleTraceWidget):

    def set_out_fig(self):

        timeseries = self.controls['timeseries'].value

        time_window = self.controls['time_window'].value

        istart = timeseries_time_to_ind(timeseries, time_window[0])
        istop = timeseries_time_to_ind(timeseries, time_window[1])

        yy, units = get_timeseries_in_units(timeseries, istart, istop)

        self.out_fig = go.FigureWidget(data=go.Scatter(
            x=get_timeseries_tt(timeseries, istart, istop),
            y=yy))

        self.out_fig.update_layout(
            title=timeseries.name,
            xaxis_title="time (s)",
            yaxis_title=units)

        def on_change(change):
            time_window = self.controls['time_window'].value
            istart = timeseries_time_to_ind(timeseries, time_window[0])
            istop = timeseries_time_to_ind(timeseries, time_window[1])

            yy, units = get_timeseries_in_units(timeseries, istart, istop)

            self.out_fig.data[0].x = get_timeseries_tt(timeseries, istart, istop)
            self.out_fig.data[0].y = yy

        self.controls['time_window'].observe(on_change)


class SeparateTracesPlotlyWidget(SingleTraceWidget):

    def set_out_fig(self):

        timeseries = self.controls['timeseries'].value

        time_window = self.controls['time_window'].value

        istart = timeseries_time_to_ind(timeseries, time_window[0])
        istop = timeseries_time_to_ind(timeseries, time_window[1])

        data, units = get_timeseries_in_units(timeseries, istart, istop)

        self.out_fig = go.FigureWidget(make_subplots(rows=data.shape[1], cols=1))

        tt = get_timeseries_tt(timeseries, istart, istop)

        for i, (yy, xyz) in enumerate(zip(data.T, ('x', 'y', 'z'))):
            self.out_fig.add_trace(
                go.Scatter(x=tt, y=yy),
                row=i + 1, col=1)
            if units:
                yaxes_label = '{} ({})'.format(xyz, units)
            else:
                yaxes_label = xyz
            self.out_fig.update_yaxes(title_text=yaxes_label, row=i+1, col=1)
        self.out_fig.update_xaxes(title_text='time (s)', row=i+1, col=1)
        self.out_fig.update_layout(showlegend=False, title=timeseries.name)

        def on_change(change):
            time_window = self.controls['time_window'].value
            istart = timeseries_time_to_ind(timeseries, time_window[0])
            istop = timeseries_time_to_ind(timeseries, time_window[1])

            tt = get_timeseries_tt(timeseries, istart, istop)
            yy, units = get_timeseries_in_units(timeseries, istart, istop)

            for i, dd in enumerate(yy.T):
                self.out_fig.data[i].x = tt
                self.out_fig.data[i].y = dd

        self.controls['time_window'].observe(on_change)


def plot_grouped_traces(time_series: TimeSeries, time_window=None, order=None, ax=None, figsize=(9.7, 7),
                        group_inds=None, labels=None, colors=color_wheel, show_legend=True, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if time_window is None:
        t_ind_start = 0
        t_ind_stop = None
    else:
        t_ind_start = timeseries_time_to_ind(time_series, time_window[0])
        t_ind_stop = timeseries_time_to_ind(time_series, time_window[1])

    tt = get_timeseries_tt(time_series, t_ind_start, t_ind_stop)

    unique_sorted_order, inverse_sort = np.unique(order, return_inverse=True)
    mini_data = time_series.data[t_ind_start:t_ind_stop, unique_sorted_order][:, inverse_sort]

    gap = np.median(np.nanstd(mini_data, axis=0)) * 20
    offsets = np.arange(len(order)) * gap

    if group_inds is not None:
        ugroup_inds = np.unique(group_inds)
        handles = []

        for i, ui in enumerate(ugroup_inds):
            color = colors[ugroup_inds[i] % len(colors)]
            lines_handle = ax.plot(tt, mini_data[:, group_inds == ui] + offsets[group_inds == ui],
                                   color=color)
            handles.append(lines_handle[0])

        if show_legend:
            ax.legend(handles=handles[::-1], labels=list(labels[ugroup_inds][::-1]), loc='upper left',
                      bbox_to_anchor=(1.01, 1))
    else:
        ax.plot(tt, mini_data + offsets, color='k')

    ax.set_xlim((tt[0], tt[-1]))
    ax.set_xlabel('time (s)')

    if len(offsets):
        ax.set_ylim(-gap, offsets[-1] + gap)
    if len(order) <= 30:
        ax.set_yticks(offsets)
        ax.set_yticklabels(order)
    else:
        ax.set_yticks([])


class BaseGroupedTraceWidget(widgets.HBox):
    def __init__(self, time_series: TimeSeries, dynamic_table_region_name, neurodata_vis_spec=None, **kwargs):
        super().__init__()
        self.time_series = time_series

        self.tmin = get_timeseries_mint(time_series)
        self.tmax = get_timeseries_maxt(time_series)
        self.time_window_controller = StartAndDurationController(tmin=self.tmin, tmax=self.tmax, start=self.tmin,
                                                                 duration=5)

        dynamic_table_region = getattr(time_series, dynamic_table_region_name)
        table = dynamic_table_region.table
        referenced_rows = dynamic_table_region.data
        discard_rows = [x for x in range(len(table)) if x not in referenced_rows]
        self.gas = GroupAndSortController(dynamic_table=table, start_discard_rows=discard_rows)

        self.controls = dict(
            time_series=widgets.fixed(self.time_series),
            time_window=self.time_window_controller,
            gas=self.gas,
        )

        out_fig = interactive_output(plot_grouped_traces, self.controls)

        self.children = [
            self.gas,
            widgets.VBox(
                children=[
                    self.time_window_controller,
                    out_fig,
                ],
                layout=widgets.Layout(width="100%")
            )
        ]

        self.layout = widgets.Layout(width="100%")









