from nwbwidgets import view
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython import display
from collections import Iterable
from pynwb import TimeSeries, ProcessingModule
from pynwb.core import NWBDataInterface
from matplotlib.pyplot import Figure
from datetime import datetime
from .utils.timeseries import (get_timeseries_tt, get_timeseries_in_units, get_timeseries_maxt, get_timeseries_mint,
                               timeseries_time_to_ind)
from .controllers import make_time_controller, make_trace_controller


def show_ts_fields(node):
    info = []
    for key in ('description', 'unit', 'resolution', 'conversion'):
        info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    return widgets.VBox(info)


def show_timeseries(node: TimeSeries, neurodata_vis_spec=None, istart=0, istop=-1, **kwargs):
    info = []
    for key in ('description', 'comments', 'unit', 'resolution', 'conversion'):
        info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    children = [widgets.VBox(info)]

    tt = get_timeseries_tt(node, istart=istart, istop=istop)
    data, unit = get_timeseries_in_units(node, istart=istart, istop=istop)

    fig, ax = plt.subplots()
    ax.plot(tt, data, **kwargs)
    ax.set_xlabel('time (s)')
    if node.unit:
        ax.set_ylabel(node.unit)

    children.append(fig2widget(fig))

    return widgets.HBox(children=children)


def show_subject(node, **kwargs):
    field_lay = widgets.Layout(max_height='40px', max_width='150px',
                               min_height='30px', min_width='70px')
    info = []
    for key, val in node.fields.items():
        lbl_key = widgets.Label(key+':', layout=field_lay)
        lbl_val = widgets.Label(str(val), layout=field_lay)
        info.append(widgets.HBox(children=[lbl_key, lbl_val]))
    vbox = widgets.VBox(info)
    return vbox


# def show_dynamic_table(node: DynamicTable, **kwargs):
def show_dynamic_table(node, **kwargs):
    out1 = widgets.Output()
    with out1:
        display.display(node.to_dataframe())
    return out1


def show_neurodata_base(node: NWBDataInterface, neurodata_vis_spec: dict):
    """
    Gets a pynwb object and returns a Vertical Box containing textual info and
    an expandable Accordion with it's children.
    """
    field_lay = widgets.Layout(max_height='40px', max_width='500px',
                               min_height='30px', min_width='180px')
    info = []         # string data type, exposed as a Text widget
    neuro_data = []   # more complex data types, also with children
    labels = []
    for key, value in node.fields.items():
        if isinstance(value, (str, datetime)):
            lbl_key = widgets.Label(key+':', layout=field_lay)
            lbl_val = widgets.Label(str(value), layout=field_lay)
            info.append(widgets.HBox(children=[lbl_key, lbl_val]))
        elif key == 'related_publications':
            pub_list = []
            for pub in value:
                pub_list.append(widgets.HTML(value="<a href=http://dx.doi.org/"+pub[4:]+">"+pub+"</a>"))
            lbl_key = widgets.Label(key+':', layout=field_lay)
            pub_list.insert(0, lbl_key)
            info.append(widgets.HBox(children=pub_list))
        elif key == 'experimenter':
            lbl_experimenter = widgets.Label('Experimenter:', layout=field_lay)
            if isinstance(value, (list, tuple)):
                lbl_names = widgets.Label(', '.join(value), layout=field_lay)
            else:
                lbl_names = widgets.Label(value, layout=field_lay)
            hbox_exp = widgets.HBox(children=[lbl_experimenter, lbl_names])
            info.append(hbox_exp)
        elif (isinstance(value, Iterable) and len(value)) or value:
            neuro_data.append(view.nwb2widget(value, neurodata_vis_spec=neurodata_vis_spec))
            labels.append(key)
    accordion = widgets.Accordion(children=neuro_data, selected_index=None)
    for i, label in enumerate(labels):
        if hasattr(node.fields[label], 'description') and node.fields[label].description:
            accordion.set_title(i, label + ': ' + node.fields[label].description)
        else:
            accordion.set_title(i, label)
    return widgets.VBox(info + [accordion])


def dict2accordion(d, neurodata_vis_spec: dict, **pass_kwargs):
    children = [widgets.HTML('Rendering...') for _ in d]
    accordion = widgets.Accordion(children=children, selected_index=None)
    for i, label in enumerate(d):
        if hasattr(d[label], 'description') and d[label].description:
            accordion.set_title(i, label + ': ' + d[label].description)
        else:
            accordion.set_title(i, label)
        accordion.set_title(i, label)

    def on_selected_index(change):
        if change.new is not None and isinstance(change.owner.children[change.new], widgets.HTML):
            children[change.new] = nwb2widget(list(d.values())[change.new], neurodata_vis_spec=neurodata_vis_spec,
                                              **pass_kwargs)
            change.owner.children = children

    accordion.observe(on_selected_index, names='selected_index')

    return accordion


def lazy_tabs(in_dict: dict, node):
    """Creates a lazy tab object where multiple visualizations can be used for a single node and are generated on the
    fly

    Parameters
    ----------
    in_dict: dict
        keys are labels for tabs and values are functions
    node: NWBDataInterface
        instance of neurodata type to visualize

    Returns
    -------
    tab: widget

    """
    tabs_spec = list(in_dict.items())

    children = [tabs_spec[0][1](node)] + [widgets.HTML('Rendering...')
                                          for _ in range(len(tabs_spec) - 1)]
    tab = widgets.Tab(children=children)
    [tab.set_title(i, label) for i, (label, _) in enumerate(tabs_spec)]

    def on_selected_index(change):
        if isinstance(change.owner.children[change.new], widgets.HTML):
            children[change.new] = vis2widget(tabs_spec[change.new][1](node))
            change.owner.children = children

    tab.observe(on_selected_index, names='selected_index')

    return tab


def nwb2widget(node,  neurodata_vis_spec: dict, **pass_kwargs):
    for ndtype in type(node).__mro__:
        if ndtype in neurodata_vis_spec:
            spec = neurodata_vis_spec[ndtype]
            if isinstance(spec, dict):
                return lazy_tabs(spec, node)
            elif callable(spec):
                return vis2widget(spec(node, neurodata_vis_spec=neurodata_vis_spec, **pass_kwargs))
    out1 = widgets.Output()
    with out1:
        print(node)
    return out1


def vis2widget(vis):
    if isinstance(vis, widgets.Widget):
        return vis
    elif isinstance(vis, plt.Figure):
        return fig2widget(vis)
    else:
        raise ValueError('unsupported vis type')


def fig2widget(fig: Figure, **kwargs):
    out = widgets.Output()
    with out:
        plt.show(fig)
    return out


def processing_module(node: ProcessingModule, neurodata_vis_spec: dict):
    return nwb2widget(node.data_interfaces, neurodata_vis_spec=neurodata_vis_spec)


def show_text_fields(node, exclude=('comments', 'interval'), **kwargs):
    info = []
    for key in node.fields:
        if key not in exclude and isinstance(key, (str, float, int)):
            info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    return widgets.VBox(info)


def plot_traces(time_series: TimeSeries, time_window, trace_window,
                title: str = None, ylabel: str = 'traces'):
    """

    Parameters
    ----------
    time_series
    time_window
    trace_window
    title
    ylabel

    Returns
    -------

    """
    if time_window[0] is None:
        t_ind_start = 0
    else:
        t_ind_start = timeseries_time_to_ind(time_series, time_window[0])
    if time_window[1] is None:
        t_ind_stop = time_series.data.shape[0]
    else:
        t_ind_stop = timeseries_time_to_ind(time_series, time_window[1])
    data = time_series.data
    tt = get_timeseries_tt(time_series, t_ind_start, t_ind_stop)
    if data.shape[1] == len(tt):  # fix of orientation is incorrect
        mini_data = data[trace_window[0]:trace_window[1], t_ind_start:t_ind_stop].T
    else:
        mini_data = data[t_ind_start:t_ind_stop, trace_window[0]:trace_window[1]]

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
                  time_window_controller=None, time_window_starting_range=None,
                  trace_controller=None, trace_starting_range=None,
                  **kwargs):

    if time_window_controller is None:
        tmax = get_timeseries_maxt(node)
        tmin = get_timeseries_mint(node)
        if time_window_starting_range is None:
            time_window_starting_range = (tmin, min(tmin+10, tmax))
        time_window_controller = make_time_controller(tmin, tmax, start_value=time_window_starting_range)
    if trace_controller is None:
        if trace_starting_range is None:
            trace_starting_range = (0, min(30, node.data.shape[1]))
        trace_controller = make_trace_controller(node.data.shape[1], start_range=trace_starting_range)

    controls = {
        'time_series': widgets.fixed(node),
        'time_window': time_window_controller.children[0],
        'trace_window': trace_controller.children[0],
    }
    controls.update({key: widgets.fixed(val) for key, val in kwargs.items()})

    out_fig = widgets.interactive_output(plot_traces, controls)

    control_widgets = widgets.HBox(children=(time_window_controller, trace_controller))
    vbox = widgets.VBox(children=[control_widgets, out_fig])

    return vbox


