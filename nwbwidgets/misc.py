import matplotlib.pyplot as plt
import numpy as np
import pynwb
from typing import Iterable
from pynwb.misc import AnnotationSeries, Units, DecompositionSeries
from ipywidgets import widgets, fixed, Layout
from matplotlib import cm
from .controllers import float_range_controller, int_range_controller, int_controller
from .utils.units import get_spike_times, get_max_spike_time, get_min_spike_time, align_by_time_intervals
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def show_annotations(annotations: AnnotationSeries, **kwargs):
    fig, ax = plt.subplots()
    ax.eventplot(annotations.timestamps, **kwargs)
    ax.set_xlabel('time (s)')
    return fig


def show_session_raster(units: Units, time_window=None, units_window=None, cmap_name='rainbow',
                        show_obs_intervals=True, color_by='id', order_by1=None, order_by2=None,
                        show_legend=True):
    """

    Parameters
    ----------
    units: pynwb.misc.Units
    time_window: [int, int]
    units_window: [int, int]
    cmap_name: str
    show_obs_intervals: bool
    order_by1: str, optional
        None: order by id
        str: order by the values of this column of the Units table
    order_by2: str, optional
        None: order by id
        str: order by the values of this column of the Units table
    color_by: str, optional
        None: all ticks are black
        'id': color by id of unit (default)
        other str: color by value in units table
    show_legend: bool
        default = True
        Does not show legend if color_by is None or 'id'.

    Returns
    -------
    matplotlib.pyplot.Figure

    """
    if time_window is None:
        time_window = [get_min_spike_time(units), get_max_spike_time(units)]

    if units_window is None:
        units_window = [0, len(units['spike_times'].data) - 1]

    num_units = units_window[1] - units_window[0] + 1
    unit_inds = np.arange(units_window[0], units_window[1] + 1)

    if order_by1 is not None:
        if order_by2 is None:
            order = np.argsort(units[order_by1][unit_inds.tolist()])
        else:
            order = np.lexsort([units[i_order_by][unit_inds.tolist()]
                                for i_order_by in (order_by1, order_by2)])
    else:
        order = unit_inds

    reduced_spike_times = [get_spike_times(units, unit, time_window) for unit in order]

    # create colormap
    cmap = cm.get_cmap(cmap_name, num_units)
    if color_by is None:
        colors = 'k'
    else:
        if color_by == 'id':
            cvals = unit_inds
        else:
            vals = [units[color_by][x] for x in order]
            if isinstance(vals[0], str):
                labels, val_index, cvals = np.unique(vals, return_index=True, return_inverse=True)
            else:
                cvals = vals
                labels, val_index = np.unique(vals, return_index=True)
        # normalize cvals
        cvals -= min(cvals)
        cvals = cvals / max(cvals)
        colors = cmap(cvals)

    # plot spike times for each unit
    fig, ax = plt.subplots(1, 1)
    ax.figure.set_size_inches(12, 6)
    ax.eventplot(reduced_spike_times, color=colors, lineoffsets=unit_inds)

    # add observation intervals
    if show_obs_intervals and 'obs_intervals' in units:
        rects = []
        for i_unit in unit_inds:
            intervals = units['obs_intervals'][i_unit]  # TODO: use bisect here
            these_obs_intervals = intervals[(intervals[:, 1] > time_window[0]) & (intervals[:, 0] < time_window[1])]
            unobs_intervals = np.c_[these_obs_intervals[:-1, 1], these_obs_intervals[1:, 0]]

            if len(these_obs_intervals):
                # handle unobserved interval on lower bound of window
                if these_obs_intervals[0, 0] > time_window[0]:
                    unobs_intervals = np.vstack(([time_window[0], these_obs_intervals[0, 0]], unobs_intervals))

                # handle unobserved interval on lower bound of window
                if these_obs_intervals[-1, 1] < time_window[1]:
                    unobs_intervals = np.vstack((unobs_intervals, [these_obs_intervals[-1, 1], time_window[1]]))
            else:
                unobs_intervals = [time_window]

            for i_interval in unobs_intervals:
                rects.append(Rectangle((i_interval[0], i_unit-.5), i_interval[1]-i_interval[0], 1))
        pc = PatchCollection(rects, color=[0.85, 0.85, 0.85])
        ax.add_collection(pc)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Unit #')
    ax.set_xlim(time_window)
    ax.set_ylim(np.array(units_window) + [-.5, .5])
    if units_window[1] - units_window[0] <= 30:
        ax.set_yticks(range(units_window[0], units_window[1] + 1))

    if color_by not in (None, 'id') and show_legend:
        ax.legend(handles=[ax.collections[x] for x in val_index],
                  labels=labels.tolist(), title=color_by)

    return fig


def robust_unique(a):
    if isinstance(a[0], pynwb.NWBContainer):
        return np.unique([x.name for x in a])
    return np.unique(a)


def raster_widget(units: Units, unit_controller=None, time_window_controller=None):
    if time_window_controller is None:
        tmin = get_min_spike_time(units)
        tmax = get_max_spike_time(units)
        time_window_controller = float_range_controller(tmin, tmax)
    if unit_controller is None:
        unit_controller = int_range_controller(len(units['spike_times'].data)-1, start_range=(0, 100))

    candidate_cols = [x for x in units.colnames
                      if not isinstance(units[x][0], Iterable) or
                      isinstance(units[x][0], str)]

    features = [x for x in candidate_cols if len(robust_unique(units[x][:])) > 1]
    color_controller = widgets.Dropdown(options=[None] + features, description='color by',
                                        layout=Layout(width='90%'))
    order_by_controller1 = widgets.Dropdown(options=[None] + features, description='order by',
                                            layout=Layout(width='90%'))
    order_by_controller2 = widgets.Dropdown(options=[None] + features, description='then order by',
                                            layout=Layout(width='90%'))

    controls = {
        'units': fixed(units),
        'time_window': time_window_controller.children[0],
        'units_window': unit_controller.children[0],
        'color_by': color_controller,
        'order_by1': order_by_controller1,
        'order_by2': order_by_controller2
    }

    out_fig = widgets.interactive_output(show_session_raster, controls)
    color_and_order_box = widgets.VBox(
        children=(color_controller, order_by_controller1, order_by_controller2),
        layout=Layout(width='250px'))
    control_widgets = widgets.HBox(children=(time_window_controller, unit_controller, color_and_order_box))
    vbox = widgets.VBox(children=[control_widgets, out_fig])
    return vbox


def show_decomposition_series(node, **kwargs):
    # Use Rendering... as a placeholder
    ntabs = 2
    children = [widgets.HTML('Rendering...') for _ in range(ntabs)]

    def on_selected_index(change):
        # Click on Traces Tab
        if change.new == 1 and isinstance(change.owner.children[1], widgets.HTML):
            widget_box = show_decomposition_traces(node)
            children[1] = widget_box
            change.owner.children = children

    field_lay = widgets.Layout(max_height='40px', max_width='500px',
                               min_height='30px', min_width='130px')
    vbox = []
    for key, val in node.fields.items():
        lbl_key = widgets.Label(key+':', layout=field_lay)
        lbl_val = widgets.Label(str(val), layout=field_lay)
        vbox.append(widgets.HBox(children=[lbl_key, lbl_val]))
        #vbox.append(widgets.Text(value=repr(value), description=key, disabled=True))
    children[0] = widgets.VBox(vbox)

    tab_nest = widgets.Tab()
    tab_nest.children = children
    tab_nest.set_title(0, 'Fields')
    tab_nest.set_title(1, 'Traces')
    tab_nest.observe(on_selected_index, names='selected_index')
    return tab_nest


def show_decomposition_traces(node: DecompositionSeries):
    # Produce figure
    def control_plot(x0, x1, ch0, ch1):
        fig, ax = plt.subplots(nrows=nBands, ncols=1, sharex=True, figsize=(14, 7))
        for bd in range(nBands):
            data = node.data[x0:x1, ch0:ch1+1, bd]
            xx = np.arange(x0, x1)
            mu_array = np.mean(data, 0)
            sd_array = np.std(data, 0)
            offset = np.mean(sd_array)*5
            yticks = [i*offset for i in range(ch1+1-ch0)]
            for i in range(ch1+1-ch0):
                ax[bd].plot(xx, data[:, i] - mu_array[i] + yticks[i])
            ax[bd].set_ylabel('Ch #', fontsize=20)
            ax[bd].set_yticks(yticks)
            ax[bd].set_yticklabels([str(i) for i in range(ch0, ch1+1)])
            ax[bd].tick_params(axis='both', which='major', labelsize=16)
        ax[bd].set_xlabel('Time [ms]', fontsize=20)
        return fig

    nSamples = node.data.shape[0]
    nChannels = node.data.shape[1]
    nBands = node.data.shape[2]
    fs = node.rate

    # Controls
    field_lay = widgets.Layout(max_height='40px', max_width='100px',
                               min_height='30px', min_width='70px')
    x0 = widgets.BoundedIntText(value=0, min=0, max=int(1000*nSamples/fs-100),
                                layout=field_lay)
    x1 = widgets.BoundedIntText(value=nSamples, min=100, max=int(1000*nSamples/fs),
                                layout=field_lay)
    ch0 = widgets.BoundedIntText(value=0, min=0, max=int(nChannels-1), layout=field_lay)
    ch1 = widgets.BoundedIntText(value=10, min=0, max=int(nChannels-1), layout=field_lay)

    controls = {
        'x0': x0,
        'x1': x1,
        'ch0': ch0,
        'ch1': ch1
    }
    out_fig = widgets.interactive_output(control_plot, controls)

    # Assemble layout box
    lbl_x = widgets.Label('Time [ms]:', layout=field_lay)
    lbl_ch = widgets.Label('Ch #:', layout=field_lay)
    lbl_blank = widgets.Label('    ', layout=field_lay)
    hbox0 = widgets.HBox(children=[lbl_x, x0, x1, lbl_blank, lbl_ch, ch0, ch1])
    vbox = widgets.VBox(children=[hbox0, out_fig])
    return vbox


def psth_widget(units: Units, unit_controller=None, after_slider=None, before_slider=None,
                trial_event_controller=None, trial_order_controller=None, trial_color_controller=None):
    """

    Parameters
    ----------
    units: pynwb.misc.Units
    unit_controller
    after_slider
    before_slider
    trial_event_controller
    trial_order_controller
    trial_color_controller

    Returns
    -------

    """

    trials = units.get_ancestor('NWBFile').trials
    if trials is None:
        return widgets.HTML('No trials present')

    control_widgets = widgets.VBox(children=[])

    if unit_controller is None:
        nunits = len(units['spike_times'].data)
        #unit_controller = int_controller(nunits)
        unit_controller = widgets.Dropdown(options=[x for x in range(nunits)],
                                           description='unit: ')
        control_widgets.children = list(control_widgets.children) + [unit_controller]

    if trial_event_controller is None:
        trial_events = ['start_time']
        if not np.all(np.isnan(trials['stop_time'].data)):
            trial_events.append('stop_time')
        trial_events += [x.name for x in trials.columns if
                         (('_time' in x.name) and (x.name not in ('start_time', 'stop_time')))]
        trial_event_controller = widgets.Dropdown(options=trial_events,
                                                  value='start_time',
                                                  description='align to: ')

        control_widgets.children = list(control_widgets.children) + [trial_event_controller]

    if trial_order_controller is None:
        trials = units.get_ancestor('NWBFile').trials
        trial_order_controller = widgets.Dropdown(options=trials.colnames,
                                                  value='start_time',
                                                  description='order by: ')
        control_widgets.children = list(control_widgets.children) + [trial_order_controller]

    if trial_color_controller is None:
        trials = units.get_ancestor('NWBFile').trials
        trial_color_controller = widgets.Dropdown(options=[''] + list(trials.colnames),
                                                  value='',
                                                  description='color by: ')
        control_widgets.children = list(control_widgets.children) + [trial_color_controller]

    if before_slider is None:
        before_slider = widgets.FloatSlider(.5, min=0, max=5., description='before (s)', continuous_update=False)
        control_widgets.children = list(control_widgets.children) + [before_slider]

    if after_slider is None:
        after_slider = widgets.FloatSlider(2., min=0, max=5., description='after (s)', continuous_update=False)
        control_widgets.children = list(control_widgets.children) + [after_slider]

    controls = {
        'units': fixed(units),
        'index': unit_controller,
        'after': after_slider,
        'before': before_slider,
        'start_label': trial_event_controller,
        'order_by': trial_order_controller,
        'color_by': trial_color_controller
    }

    out_fig = widgets.interactive_output(trials_psth, controls)
    vbox = widgets.VBox(children=[control_widgets, out_fig])
    return vbox


def trials_psth(units: pynwb.misc.Units, index=0, start_label='start_time', before=0., after=1., order_by='start_time',
                color_by=None, cmap_name='gist_rainbow', trials_select=None):
    """

    Parameters
    ----------
    units: pynwb.misc.Units
    index: int
    start_label
    before
    after
    order_by
    color_by
    cmap_name
    trials_select

    Returns
    -------

    """
    trials = units.get_ancestor('NWBFile').trials

    data = align_by_time_intervals(units, index, trials, start_label, start_label, before, after, trials_select)

    if trials_select is None:
        order = np.argsort(trials[order_by].data[:])
    else:
        order = np.argsort(trials[order_by].data[trials_select])

    data = np.array(data)[order]

    cmap = cm.get_cmap(cmap_name)

    labels = None
    if color_by:
        coldata = trials[color_by].data[:]
        if len(np.unique(coldata)) < 5:
            labels, cvals = np.unique(coldata, return_inverse=True)
        elif np.all(np.isreal(data)):
            cvals = coldata
        else:
            coltype = 'unknown'
            cvals = np.array([0])
        cvals = cvals - min(cvals)
        cvals = cvals / max(cvals)
        cvals = cvals[order]
        colors = cmap(cvals)
        cval_inds = np.hstack((0, np.where(np.diff(cvals))[0] + 1))
    else:
        colors = 'k'

    fig, ax = plt.subplots()
    if labels is not None:
        ax = show_psth(data, colors, ax, before, after,
                       labels=labels, cval_inds=cval_inds)
    else:
        ax = show_psth(data, colors, ax, before, after)
    ax.set_title('PSTH for unit {}'.format(index))
    return fig


def show_psth(data, colors, ax, before, after, labels=None, cval_inds=None):
    event_collection = ax.eventplot(data, orientation='horizontal', colors=colors)
    ax.set_xlim((-before, after))
    ax.set_ylim(-.5, len(data)-.5)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('trials')
    ax.axvline(color=[.5, .5, .5])

    if labels is not None:
        handles = [event_collection[x] for x in cval_inds]
        ax.legend(handles=handles, labels=list(labels), bbox_to_anchor=(1.0, .6, .4, .4),
                  mode="expand", borderaxespad=0.)

    return ax

