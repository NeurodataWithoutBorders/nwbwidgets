import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets, fixed
from pynwb import TimeSeries
from .utils.timeseries import (get_timeseries_tt, get_timeseries_maxt, get_timeseries_mint,
                               timeseries_time_to_ind, get_timeseries_in_units)
from abc import abstractmethod
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .controllers import StartAndDurationController,  GroupAndSortController
from .utils.widgets import interactive_output
from .utils.plotly import multi_trace

color_wheel = plt.rcParams['axes.prop_cycle'].by_key()['color']


def show_ts_fields(node):
    info = []
    for key in ('description', 'unit', 'resolution', 'conversion'):
        info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    return widgets.VBox(info)


def show_timeseries_mpl(time_series: TimeSeries, time_window=None, ax=None,  zero_start=False, xlabel=None, ylabel=None,
                        title=None, figsize=None, **kwargs):
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

    return show_indexed_timeseries_mpl(time_series, istart=istart, istop=istop, ax=ax,  zero_start=zero_start,
                                       xlabel=xlabel, ylabel=ylabel, title=title, figsize=figsize, **kwargs)


def show_indexed_timeseries_mpl(node: TimeSeries, istart=0, istop=None, ax=None,
                                zero_start=False, xlabel='time (s)', ylabel=None, title=None, figsize=None,
                                neurodata_vis_spec=None, **kwargs):

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
        return BaseGroupedTraceWidget(node, **kwargs)
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

    mpl_plotter = show_timeseries

    def set_children(self):
        if self.foreign_time_window_controller:
            self.children = [self.out_fig]
        else:
            self.children = [self.time_window_controller, self.out_fig]


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


def _prep_timeseries(time_series, time_window=None, order=None):
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
    else:
        mini_data = time_series.data[t_ind_start:t_ind_stop]

    gap = np.median(np.nanstd(mini_data, axis=0)) * 20
    offsets = np.arange(len(order)) * gap

    mini_data = mini_data + offsets

    return mini_data, tt, offsets


def plot_grouped_traces(time_series: TimeSeries, time_window=None, order=None, ax=None, figsize=(9.7, 7),
                        group_inds=None, labels=None, colors=color_wheel, show_legend=True, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if order is None:
        order = np.arange(time_series.data.shape[1])

    mini_data, tt, offsets = _prep_timeseries(time_series, time_window, order)

    if group_inds is not None:
        ugroup_inds = np.unique(group_inds)
        handles = []

        for i, ui in enumerate(ugroup_inds):
            color = colors[ugroup_inds[i] % len(colors)]
            lines_handle = ax.plot(tt, mini_data[:, group_inds == ui],
                                   color=color)
            handles.append(lines_handle[0])

        if show_legend:
            ax.legend(handles=handles[::-1], labels=list(labels[ugroup_inds][::-1]), loc='upper left',
                      bbox_to_anchor=(1.01, 1))
    else:
        ax.plot(tt, mini_data, color='k')

    ax.set_xlim((tt[0], tt[-1]))
    ax.set_xlabel('time (s)')

    if len(offsets):
        ax.set_ylim(-offsets[0], offsets[-1] + offsets[0])
    if len(order) <= 30:
        ax.set_yticks(offsets)
        ax.set_yticklabels(order)
    else:
        ax.set_yticks([])


def plot_grouped_traces_plotly(time_series: TimeSeries, time_window, order, group_inds=None, labels=None,
                               colors=color_wheel, fig=None, **kwargs):
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
        multi_trace(tt, mini_data.T, 'black', fig=fig)
    fig.update_layout(
        title=time_series.name,
        xaxis_title="time (s)")
    fig.update_yaxes(tickvals=list(offsets), ticktext=[str(i) for i in order])

    return fig


class BaseGroupedTraceWidget(widgets.HBox):
    def __init__(self, time_series: TimeSeries, dynamic_table_region_name=None,
                 foreign_time_window_controller: StartAndDurationController = None,
                 foreign_group_and_sort_controller: GroupAndSortController = None,
                 mpl_plotter=plot_grouped_traces, **kwargs):
        """

        Parameters
        ----------
        time_series: TimeSeries
        dynamic_table_region_name: str, optional
        foreign_time_window_controller: StartAndDurationController, optional
        foreign_group_and_sort_controller: GroupAndSortController, optional
        kwargs
        """

        if dynamic_table_region_name is not None and foreign_group_and_sort_controller is not None:
            raise TypeError('You cannot supply both `dynamic_table_region_name` and `foreign_group_and_sort_controller`.')

        super().__init__()
        self.time_series = time_series

        if foreign_time_window_controller is not None:
            self.time_window_controller = foreign_time_window_controller
        else:
            self.tmin = get_timeseries_mint(time_series)
            self.tmax = get_timeseries_maxt(time_series)
            self.time_window_controller = StartAndDurationController(tmin=self.tmin, tmax=self.tmax, start=self.tmin,
                                                                     duration=5)

        self.controls = dict(
            time_series=widgets.fixed(self.time_series),
            time_window=self.time_window_controller
        )
        if foreign_group_and_sort_controller is None:
            if dynamic_table_region_name is not None:
                dynamic_table_region = getattr(time_series, dynamic_table_region_name)
                table = dynamic_table_region.table
                referenced_rows = dynamic_table_region.data
                discard_rows = [x for x in range(len(table)) if x not in referenced_rows]
                self.gas = GroupAndSortController(dynamic_table=table, start_discard_rows=discard_rows)
                self.controls.update(gas=self.gas)
            else:
                self.gas = None
        else:
            self.gas = foreign_group_and_sort_controller
            self.controls.update(gas=self.gas)

        out_fig = interactive_output(mpl_plotter, self.controls)

        if foreign_time_window_controller:
            right_panel = out_fig
        else:
            right_panel = widgets.VBox(
                children=[
                    self.time_window_controller,
                    out_fig,
                ],
                layout=widgets.Layout(width="100%")
            )

        if foreign_group_and_sort_controller or self.gas is None:
            self.children = [right_panel]
        else:

            self.children = [
                self.gas,
                right_panel
            ]

        self.layout = widgets.Layout(width="100%")


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
        self.time_window_controller = StartAndDurationController(tmin=self.tmin, tmax=self.tmax, start=self.tmin,
                                                                 duration=5)

        widgets = [widget_class(time_series, foreign_time_window_controller=self.time_window_controller)
                   for widget_class, time_series in zip(widget_class_list, time_series_list)]
        self.children = [self.time_window_controller] + widgets
