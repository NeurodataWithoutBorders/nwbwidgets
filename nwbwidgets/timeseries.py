import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets, fixed
from pynwb import TimeSeries
import pynwb
from .utils.timeseries import (get_timeseries_tt, get_timeseries_maxt, get_timeseries_mint,
                               timeseries_time_to_ind, get_timeseries_in_units)
from .controllers import make_time_window_controller,  RangeController
from .base import fig2widget


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


def show_timeseries(node: TimeSeries, neurodata_vis_spec=None, istart=0, istop=None, ax=None, zero_start=False,
                    **kwargs):
    fig = show_timeseries_mpl(node, neurodata_vis_spec, istart, istop, ax, zero_start, **kwargs)

    info = []
    for key in ('description', 'comments', 'unit', 'resolution', 'conversion'):
        info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    children = [widgets.VBox(info)]

    children.append(fig2widget(fig))

    return widgets.HBox(children=children)


def plot_traces(time_series: TimeSeries, time_start=0, time_duration=None, trace_window=None,
                title: str = None, ylabel: str = 'traces'):
    """

    Parameters
    ----------
    time_series: pynwb.TimeSeries
    time_start: float
        Start time in seconds
    time_duration: float, optional
        Duration in seconds. Default:
    trace_window: [int int], optional
        Index range of traces to view
    title: str, optional
    ylabel: str, optional

    Returns
    -------

    """

    if time_start == 0:
        t_ind_start = 0
    else:
        t_ind_start = timeseries_time_to_ind(time_series, time_start)
    if time_duration is None:
        t_ind_stop = None
    else:
        t_ind_stop = timeseries_time_to_ind(time_series, time_start + time_duration)

    if trace_window is None:
        trace_window = [0, time_series.data.shape[1]]
    tt = get_timeseries_tt(time_series, t_ind_start, t_ind_stop)
    if time_series.data.shape[1] == len(tt):  # fix of orientation is incorrect
        mini_data = time_series.data[trace_window[0]:trace_window[1], t_ind_start:t_ind_stop].T
    else:
        mini_data = time_series.data[t_ind_start:t_ind_stop, trace_window[0]:trace_window[1]]

    gap = np.median(np.nanstd(mini_data, axis=0)) * 20
    offsets = np.arange(trace_window[1] - trace_window[0]) * gap

    fig, ax = plt.subplots()
    ax.figure.set_size_inches(12, 6)
    ax.plot(tt, mini_data + offsets)
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


def traces_widget(node: TimeSeries, neurodata_vis_spec: dict = None,
                  time_window_controller=None, start=None, dur=None,
                  trace_controller=None, trace_starting_range=None,
                  **kwargs):

    if time_window_controller is None:
        tmax = get_timeseries_maxt(node)
        tmin = get_timeseries_mint(node)
        if start is None:
            start = tmin
        if dur is None:
            dur = min(tmax-tmin, 5)
        time_window_controller = make_time_window_controller(tmin, tmax, start=start, duration=dur)
    if trace_controller is None:
        if trace_starting_range is None:
            trace_starting_range = (0, min(30, node.data.shape[1]))
        trace_controller = RangeController(0, node.data.shape[1], start_range=trace_starting_range,
                                           description='channels', dtype='int', orientation='vertical')

    controls = {
        'time_series': widgets.fixed(node),
        'time_start': time_window_controller.children[0],
        'time_duration': time_window_controller.children[1],
        'trace_window': trace_controller.slider,
    }
    controls.update({key: widgets.fixed(val) for key, val in kwargs.items()})

    out_fig = widgets.interactive_output(plot_traces, controls)

    lower = widgets.HBox(children=[
        trace_controller,
        out_fig
    ])

    out = widgets.VBox(children=[
        time_window_controller,
        lower
    ])

    return out


def single_trace_widget(timeseries: TimeSeries, time_window_controller=None):

    controls = dict(timeseries=fixed(timeseries))

    gen_time_window_controller = False
    if time_window_controller is None:
        gen_time_window_controller = True
        tmin = get_timeseries_mint(timeseries)
        tmax = get_timeseries_maxt(timeseries)
        time_window_controller = RangeController(tmin, tmax, start_value=[tmin, min(tmin+30, tmax)])

    controls.update(time_window=time_window_controller.slider)

    out_fig = widgets.interactive_output(show_trace, controls)

    if gen_time_window_controller:
        return widgets.VBox(children=[time_window_controller, out_fig])
    else:
        return widgets.VBox(children=[out_fig])


def show_trace(timeseries, time_window):
    istart = timeseries_time_to_ind(timeseries, time_window[0])
    istop = timeseries_time_to_ind(timeseries, time_window[1])

    return show_timeseries_mpl(timeseries, istart=istart, istop=istop).get_figure()