from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import scipy
from ipywidgets import widgets, fixed, Layout
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from pynwb.misc import AnnotationSeries, Units, DecompositionSeries

from .controllers import make_trial_event_controller, int_controller, RangeController
from .utils.dynamictable import group_and_sort, infer_categorical_columns
from .utils.units import get_spike_times, get_max_spike_time, get_min_spike_time, align_by_time_intervals, \
    get_unobserved_intervals
from .utils.mpl import create_big_ax
from .utils.pynwb import robust_unique
from .utils.widgets import interactive_output
from .analysis.spikes import compute_smoothed_firing_rate

color_wheel = plt.rcParams['axes.prop_cycle'].by_key()['color']


def show_annotations(annotations: AnnotationSeries, **kwargs):
    fig, ax = plt.subplots()
    ax.eventplot(annotations.timestamps, **kwargs)
    ax.set_xlabel('time (s)')
    return fig


def show_session_raster(units: Units, time_window=None, units_select=(), units_window=None, show_obs_intervals=True,
                        group_vals=None, order_vals=None, show_legend=True, limit=None):
    """

    Parameters
    ----------
    units: pynwb.misc.Units
    time_window: [int, int]
    units_select: list(int)
    units_window: [int, int]
    show_obs_intervals: bool
    group_vals: array-like, optional
    order_vals: array-like, optional
    show_legend: bool
        default = True
        Does not show legend if color_by is None or 'id'.
    limit: int, optional

    Returns
    -------
    matplotlib.pyplot.Figure

    """

    if time_window is None:
        time_window = [get_min_spike_time(units), get_max_spike_time(units)]

    if units_window is None:
        units_window = [0, len(units)]

    if group_vals is None and order_vals is None:
        order, group_inds, labels = np.arange(units_window[0], units_window[1], dtype='int'), None, None
    else:
        order, group_inds, labels = group_and_sort(group_vals=group_vals, order_vals=order_vals, window=units_window,
                                                   limit=limit)

    if not units_select == ():
        order = units_select[order]

    data = [get_spike_times(units, unit, time_window) for unit in order]

    if show_obs_intervals:
        unobserved_intervals_list = get_unobserved_intervals(units, time_window, order)
    else:
        unobserved_intervals_list = None

    ax = plot_grouped_events(data, time_window, group_inds=group_inds, labels=labels, show_legend=show_legend,
                             offset=units_window[0], unobserved_intervals_list=unobserved_intervals_list)
    ax.set_ylabel('unit #')

    return ax


class RasterWidget(widgets.HBox):
    def __init__(self, units: Units, units_window_controller=None, time_window_controller=None):
        super(RasterWidget, self).__init__()

        self.units = units

        self.controls = dict(units=fixed(units))
        if time_window_controller is None:
            self.tmin = get_min_spike_time(units)
            self.tmax = get_max_spike_time(units)
            self.time_window_controller = RangeController(self.tmin, self.tmax,
                                                          start_value=[self.tmin, min(self.tmin + 30, self.tmax)])
        else:
            self.time_window_controller = time_window_controller
            self.tmin = self.time_window_controller.vmin
            self.tmax = self.time_window_controller.vmax
        self.controls.update(time_window=self.time_window_controller)

        if units_window_controller is None:
            self.nunits = len(units['spike_times'].data) - 1
            self.units_window_controller = RangeController(0, self.nunits, start_value=(0, min(100, self.nunits)),
                                                           dtype='int', orientation='vertical', description='units')
        else:
            self.units_window_controller = units_window_controller
            self.nunits = self.units_window_controller.vmax

        self.controls.update(units_window=self.units_window_controller)

        groups = self.get_groups()

        group_controller = widgets.Dropdown(options=[None] + list(groups), description='group by',
                                            layout=Layout(width='90%'),
                                            style={'description_width': 'initial'})
        self.controls.update(group_by=group_controller)

        limit_controller = widgets.BoundedIntText(value=50, min=-1, max=99999, description='limit',
                                                  layout=Layout(width='90%'),
                                                  style={'description_width': 'initial'},
                                                  disabled=True)

        def set_max_window(group_by, limit):
            group_vals = self.get_group_vals(self.units, group_by)
            if group_vals.dtype == np.float64:
                group_vals = group_vals[~np.isnan(group_vals)]
            nunits = sum(min(sum(group_vals == x), limit) for x in np.unique(group_vals))
            self.units_window_controller.slider.max = nunits

        def group_disable_limit(change):
            if change['name'] == 'label':
                if change['new'] in ('None', '', None):
                    limit_controller.disabled = True
                else:
                    limit_controller.disabled = False

        def group_by_set_max_window(change):
            if change['name'] == 'label':
                if change['new'] in ('None', '', None):
                    self.units_window_controller.slider.max = len(units)
                else:
                    set_max_window(change['new'], limit_controller.value)

        def limit_set_max_window(change):
            if change['name'] == 'value':
                set_max_window(group_controller.value, change['new'])

        group_controller.observe(group_disable_limit)
        group_controller.observe(group_by_set_max_window)
        limit_controller.observe(limit_set_max_window)

        orderable_features = self.get_orderable_cols()

        order_by_controller = widgets.Dropdown(options=[None] + orderable_features, description='order by',
                                               layout=Layout(width='90%'),
                                               style={'description_width': 'initial'})

        self.controls.update(order_by=order_by_controller, limit=limit_controller)

        out_fig = interactive_output(show_session_raster, self.controls, self.process_controls)

        dropdown_box = widgets.VBox(children=(group_controller,
                                              limit_controller,
                                              order_by_controller),
                                    layout=Layout(width='150px'))

        self.children = [
                widgets.VBox(children=[dropdown_box, self.units_window_controller]),
                widgets.VBox(children=[self.time_window_controller, out_fig])
            ]

    def get_groups(self):
        return infer_categorical_columns(self.units)

    @staticmethod
    def get_group_vals(dynamic_table, group_by, units_select=()):
        if group_by is None:
            return None
        elif group_by in dynamic_table:
            return dynamic_table[group_by][:][units_select]

    def get_orderable_cols(self):
        candidate_cols = [x for x in self.units.colnames
                          if not isinstance(self.units[x][0], Iterable) or
                          isinstance(self.units[x][0], str)]
        return [x for x in candidate_cols if len(robust_unique(self.units[x][:])) > 1]

    def get_trials_select(self):
        return ()  # all trials

    def process_controls(self, control_states):
        order_by = control_states.pop('order_by')
        control_states['order_vals'] = self.get_group_vals(self.units, order_by, self.get_trials_select())

        group_by = control_states.pop('group_by')
        control_states['group_vals'] = self.get_group_vals(self.units, group_by, self.get_trials_select())

        return control_states


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
            offset = np.mean(sd_array) * 5
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


class PSTHWidget(widgets.VBox):
    def __init__(self, units: Units, unit_index=0, unit_controller=None, sigma_in_secs=.05, ntt=1000):

        self.units = units

        super(PSTHWidget, self).__init__()

        self.trials = self.get_trials()
        if self.trials is None:
            self.children = [widgets.HTML('No trials present')]
            return

        if unit_controller is None:
            nunits = len(units['spike_times'].data)
            unit_controller = widgets.Dropdown(options=[x for x in range(nunits)], value=unit_index, description='unit')

        trial_event_controller = make_trial_event_controller(self.trials)
        trial_order_controller = widgets.Dropdown(options=self.trials.colnames, value='start_time',
                                                  description='order by')
        trial_group_controller = widgets.Dropdown(options=[None] + list(self.trials.colnames), description='group by')
        before_slider = widgets.FloatSlider(.5, min=0, max=5., description='before (s)', continuous_update=False)
        after_slider = widgets.FloatSlider(2., min=0, max=5., description='after (s)', continuous_update=False)

        self.children = [
            unit_controller,
            trial_event_controller,
            trial_order_controller,
            trial_group_controller,
            before_slider,
            after_slider]

        self.controls = {
            'units': fixed(units),
            'sigma_in_secs': fixed(sigma_in_secs),
            'ntt': fixed(ntt),
            'index': unit_controller,
            'after': after_slider,
            'before': before_slider,
            'start_label': trial_event_controller,
            'order_by': trial_order_controller,
            'group_by': trial_group_controller
        }

        self.select_trials()

        out_fig = interactive_output(trials_psth, self.controls, process_controls=self.process_controls)

        self.children = list(self.children) + [out_fig]

    def get_trials(self):
        return self.units.get_ancestor('NWBFile').trials

    def select_trials(self):
        return

    @staticmethod
    def get_group_vals(dynamic_table, group_by):
        if group_by is None:
            return None
        elif group_by in dynamic_table:
            return dynamic_table[group_by][:]
        else:
            raise ValueError('{} not found in trials'.format(group_by))

    def process_controls(self, control_states):
        order_by = control_states.pop('order_by')
        control_states['order_vals'] = np.array(self.get_group_vals(self.trials, order_by))

        group_by = control_states.pop('group_by')
        control_states['group_vals'] = np.array(self.get_group_vals(self.trials, group_by))

        return control_states


def trials_psth(units: pynwb.misc.Units, index, start_label='start_time',
                before=0., after=1., order_vals=None, group_vals=None, trials_select=(),
                sigma_in_secs=0.05, ntt=1000):
    """

    Parameters
    ----------
    units: pynwb.misc.Units
    index: int
        Index of unit
    start_label: str, optional
        Trial column name to align on
    before: float
        Time before that event (should be positive)
    after: float
        Time after that event
    order_vals: array-like, optional
        How to order trials
    group_vals: array-like, optional
        How to group trials
    sigma_in_secs: float, optional
        standard deviation of gaussian kernel
    ntt:
        Number of time points to use for smooth curve

    Returns
    -------
    matplotlib.Figure

    """
    trials = units.get_ancestor('NWBFile').trials
    if trials is None:
        trials = units.get_ancestor('NWBFile').epochs

    if group_vals is not None and group_vals.dtype == np.float64:
        if trials_select == ():
            trials_select = np.ones((len(group_vals),), dtype='bool')
        trials_select &= ~np.isnan(group_vals)

    if group_vals is None and order_vals is None:
        order, group_inds, labels = np.where(trials_select)[0], None, None
    else:
        if group_vals is not None:
            group_vals = group_vals[trials_select]
        if order_vals is not None:
            order_vals = order_vals[trials_select]
        order, group_inds, labels = group_and_sort(group_vals=group_vals, order_vals=order_vals)
        if not trials_select == ():
            order = np.where(trials_select)[0][order]

    data = align_by_time_intervals(units, index, trials, start_label, start_label, before, after, order)
    # expanded data so that gaussian smoother uses larger window than is viewed
    expanded_data = align_by_time_intervals(units, index, trials, start_label, start_label,
                                            before + sigma_in_secs * 4,
                                            after + sigma_in_secs * 4,
                                            order)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    show_psth_raster(data, before, after, group_inds, labels, ax=axs[0])

    axs[0].set_title('PSTH for unit {}'.format(index))
    axs[0].set_xticks([])
    axs[0].set_xlabel('')

    show_psth_smoothed(expanded_data, axs[1], before, after, group_inds,
                       sigma_in_secs=sigma_in_secs, ntt=ntt)
    return fig


def show_psth_smoothed(data, ax, before, after, group_inds=None, sigma_in_secs=.05, ntt=1000,
                       align_line_color=(.7, .7, .7)):

    all_data = np.hstack(data)
    if not len(all_data):
        return
    tt = np.linspace(min(all_data), max(all_data), ntt)
    smoothed = np.array([compute_smoothed_firing_rate(x, tt, sigma_in_secs) for x in data])

    if group_inds is None:
        group_inds = np.zeros((len(smoothed)))
    group_stats = []
    for group in range(len(np.unique(group_inds))):
        this_mean = np.mean(smoothed[group_inds == group], axis=0)
        err = scipy.stats.sem(smoothed[group_inds == group], axis=0)
        group_stats.append({'mean': this_mean,
                            'lower': this_mean - 2 * err,
                            'upper': this_mean + 2 * err})
    for stats, color in zip(group_stats, color_wheel):
        ax.plot(tt, stats['mean'], color=color)
        ax.fill_between(tt, stats['lower'], stats['upper'], alpha=.2, color=color)
    ax.set_xlim([-before, after])
    ax.set_ylabel('firing rate (Hz)')
    ax.set_xlabel('time (s)')

    ax.axvline(color=align_line_color)


def plot_grouped_events(data, window, group_inds=None, colors=color_wheel, ax=None, labels=None,
                        show_legend=True, offset=0, unobserved_intervals_list=None):
    data = np.asarray(data)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    if group_inds is not None:
        ugroup_inds = np.unique(group_inds)
        handles = []
        for i, ui in enumerate(ugroup_inds):
            color = colors[ugroup_inds[i] % len(colors)]
            lineoffsets = np.where(group_inds == ui)[0] + offset
            event_collection = ax.eventplot(data[group_inds == ui],
                                            orientation='horizontal',
                                            lineoffsets=lineoffsets,
                                            color=color)
            handles.append(event_collection[0])
        if show_legend:
            ax.legend(handles=handles[::-1], labels=list(labels[ugroup_inds][::-1]), loc='upper left',
                      bbox_to_anchor=(1.01, 1))
    else:
        ax.eventplot(data, orientation='horizontal', color='k', lineoffsets=np.arange(len(data)) + offset)

    if unobserved_intervals_list is not None:
        plot_unobserved_intervals(unobserved_intervals_list, ax, offset=offset)

    ax.set_xlim(window)
    ax.set_xlabel('time (s)')
    ax.set_ylim(np.array([-.5, len(data) - .5]) + offset)
    if len(data) <= 30:
        ax.set_yticks(range(offset, len(data) + offset))

    return ax


def plot_unobserved_intervals(unobserved_intervals_list, ax, offset=0, color=(0.85, 0.85, 0.85)):
    for irow, unobs_intervals in enumerate(unobserved_intervals_list):
        rects = [Rectangle((i_interval[0], irow - .5 + offset),
                           i_interval[1] - i_interval[0], 1)
                 for i_interval in unobs_intervals]
        pc = PatchCollection(rects, color=color)
        ax.add_collection(pc)


def show_psth_raster(data, before=0.5, after=2.0, group_inds=None, labels=None, ax=None, show_legend=True,
                     align_line_color=(0.7, 0.7, 0.7)):
    if not len(data):
        return ax
    ax = plot_grouped_events(data, [-before, after], group_inds, color_wheel, ax, labels,
                             show_legend=show_legend)
    ax.set_ylabel('trials')
    ax.axvline(color=align_line_color)
    return ax


def raster_grid(units: pynwb.misc.Units, time_intervals: pynwb.epoch.TimeIntervals, index, before, after,
                rows_label=None, cols_label=None, trials_select=None, align_by='start_time'):
    """

    Parameters
    ----------
    units: pynwb.misc.Units
    time_intervals: pynwb.epoch.TimeIntervals
    index: int
    before: float
    after: float
    rows_label: str
    cols_label: str
    trials_select: np.array(dtype=bool)
    align_by: str

    Returns
    -------

    """
    if time_intervals is None:
        raise ValueError('trials must exist (trials cannot be None)')

    if trials_select is None:
        trials_select = np.ones((len(time_intervals),)).astype('bool')

    if rows_label is not None:
        row_vals = time_intervals[rows_label][:]
        urow_vals = np.unique(row_vals[trials_select])
        if urow_vals.dtype == np.float64:
            urow_vals = urow_vals[~np.isnan(urow_vals)]

    else:
        urow_vals = [None]
    nrows = len(urow_vals)

    if cols_label is not None:
        col_vals = time_intervals[cols_label][:]
        ucol_vals = np.unique(col_vals[trials_select])
        if ucol_vals.dtype == np.float64:
            ucol_vals = ucol_vals[~np.isnan(ucol_vals)]

    else:
        ucol_vals = [None]
    ncols = len(ucol_vals)

    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False, figsize=(10, 10))
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
                data = align_by_time_intervals(units, index, time_intervals, align_by, align_by,
                                               before, after, ax_trials_select)
                show_psth_raster(data, before, after, ax=ax)
                ax.set_xlabel('')
                ax.set_ylabel('')
                if ax.is_first_col():
                    ax.set_ylabel(row)
                if ax.is_last_row():
                    ax.set_xlabel(col)

    big_ax.set_xlabel(cols_label, labelpad=50)
    big_ax.set_ylabel(rows_label, labelpad=60)

    return fig


class RasterGridWidget(widgets.VBox):

    def __init__(self, units: Units, unit_index=0):
        super(RasterGridWidget, self).__init__()

        self.units = units

        self.trials = self.get_trials()
        if self.trials is None:
            self.children = [widgets.HTML('No trials present')]
            return

        groups = list(self.trials.colnames)

        rows_controller = widgets.Dropdown(options=[None] + list(groups), description='rows')
        cols_controller = widgets.Dropdown(options=[None] + list(groups), description='cols')

        trial_event_controller = make_trial_event_controller(self.trials)
        unit_controller = int_controller(len(units['spike_times'].data) - 1, value=unit_index)

        before_slider = widgets.FloatSlider(.1, min=0, max=5., description='before (s)', continuous_update=False)
        after_slider = widgets.FloatSlider(1., min=0, max=5., description='after (s)', continuous_update=False)

        self.controls = {
            'units': fixed(units),
            'time_intervals': fixed(self.trials),
            'index': unit_controller.children[0],
            'after': after_slider,
            'before': before_slider,
            'align_by': trial_event_controller,
            'rows_label': rows_controller,
            'cols_label': cols_controller
        }

        self.children = [
            unit_controller,
            rows_controller,
            cols_controller,
            trial_event_controller,
            before_slider,
            after_slider,
        ]

        self.select_trials()

        out_fig = interactive_output(raster_grid, self.controls, self.process_controls)

        self.children = list(self.children) + [out_fig]

    def get_groups(self):
        return infer_categorical_columns(self.trials)

    @staticmethod
    def get_group_vals(dynamic_table, group_by, units_select=()):
        if group_by is None:
            return None
        elif group_by in dynamic_table:
            return dynamic_table[group_by][:][units_select]
        else:
            raise ValueError('{} not found in trials'.format(group_by))

    def get_trials(self):
        return self.units.get_ancestor('NWBFile').trials

    def select_trials(self):
        return

    def process_controls(self, control_states):
        return control_states
