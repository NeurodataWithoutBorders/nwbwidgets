from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import scipy
from ipywidgets import widgets, fixed, Layout
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from pynwb.misc import AnnotationSeries, Units, DecompositionSeries

from .controllers import float_range_controller, int_range_controller, make_trial_event_controller, int_controller
from .utils.dynamictable import get_group_inds_and_order,infer_categorical_columns
from .utils.units import get_spike_times, get_max_spike_time, get_min_spike_time, align_by_time_intervals
from .utils.mpl import create_big_ax

color_wheel = plt.rcParams['axes.prop_cycle'].by_key()['color']


def show_annotations(annotations: AnnotationSeries, **kwargs):
    fig, ax = plt.subplots()
    ax.eventplot(annotations.timestamps, **kwargs)
    ax.set_xlabel('time (s)')
    return fig


def show_session_raster(units: Units, time_window=None, units_window=None, show_obs_intervals=True,
                        group_by=None, order_by=None, show_legend=True):
    """

    Parameters
    ----------
    units: pynwb.misc.Units
    time_window: [int, int]
    units_window: [int, int]
    show_obs_intervals: bool
    group_by: str, optional
    order_by: str, optional
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

    unit_inds = np.arange(units_window[0], units_window[1] + 1)

    order, group_inds, labels = get_group_inds_and_order(units, rows=unit_inds,
                                                         group_by=group_by, order_by=order_by)

    data = [get_spike_times(units, unit, time_window) for unit in unit_inds[order]]

    # plot spike times for each unit
    fig, ax = plt.subplots(1, 1)
    ax.figure.set_size_inches(12, 6)
    plot_grouped_events(data, time_window, group_inds=group_inds, labels=labels, show_legend=show_legend, ax=ax)

    # add observation intervals
    if show_obs_intervals and 'obs_intervals' in units:
        rects = []
        for i_unit in unit_inds:
            intervals = units['obs_intervals'][i_unit]  # TODO: use bisect here
            intervals = np.array(intervals)
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

    ax.set_xlabel('time (s)')
    ax.set_ylabel('unit #')
    ax.set_xlim(time_window)
    ax.set_ylim(np.array(units_window) + [-.5, .5])
    if units_window[1] - units_window[0] <= 30:
        ax.set_yticks(range(units_window[0], units_window[1] + 1))

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

    groups = infer_categorical_columns(units)
    group_controller = widgets.Dropdown(options=[None] + list(groups), description='color by',
                                        layout=Layout(width='90%'))

    features = [x for x in candidate_cols if len(robust_unique(units[x][:])) > 1]
    order_by_controller = widgets.Dropdown(options=[None] + features, description='order by',
                                           layout=Layout(width='90%'))

    controls = {
        'units': fixed(units),
        'time_window': time_window_controller.children[0],
        'units_window': unit_controller.children[0],
        'group_by': group_controller,
        'order_by': order_by_controller,
    }

    out_fig = widgets.interactive_output(show_session_raster, controls)
    color_and_order_box = widgets.VBox(
        children=(group_controller, order_by_controller),
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
                trial_event_controller=None, trial_order_controller=None,
                trial_group_controller=None, sigma_in_secs=.05, ntt=1000):
    """

    Parameters
    ----------
    units: pynwb.misc.Units
    unit_controller
    after_slider
    before_slider
    trial_event_controller
    trial_order_controller
    trial_group_controller
    sigma_in_secs: float
    ntt: int
        Number of timepoints to use for smoothed PSTH

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
        trial_event_controller = make_trial_event_controller(trials)
        control_widgets.children = list(control_widgets.children) + [trial_event_controller]

    if trial_order_controller is None:
        trials = units.get_ancestor('NWBFile').trials
        trial_order_controller = widgets.Dropdown(options=trials.colnames,
                                                  value='start_time',
                                                  description='order by: ')
        control_widgets.children = list(control_widgets.children) + [trial_order_controller]

    if trial_group_controller is None:
        trials = units.get_ancestor('NWBFile').trials
        trial_group_controller = widgets.Dropdown(options=[None] + list(trials.colnames),
                                                  description='group by')
        control_widgets.children = list(control_widgets.children) + [trial_group_controller]

    if before_slider is None:
        before_slider = widgets.FloatSlider(.5, min=0, max=5., description='before (s)', continuous_update=False)
        control_widgets.children = list(control_widgets.children) + [before_slider]

    if after_slider is None:
        after_slider = widgets.FloatSlider(2., min=0, max=5., description='after (s)', continuous_update=False)
        control_widgets.children = list(control_widgets.children) + [after_slider]

    controls = {
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

    out_fig = widgets.interactive_output(trials_psth, controls)
    vbox = widgets.VBox(children=[control_widgets, out_fig])
    return vbox


def trials_psth(units: pynwb.misc.Units, index=0, start_label='start_time',
                before=0., after=1., order_by=None, group_by=None, trials_select=None,
                sigma_in_secs=0.05, ntt=1000):
    """

    Parameters
    ----------
    units: pynwb.misc.Units
    index: int
    start_label
    before
    after
    order_by
    group_by
    trials_select

    Returns
    -------

    """
    trials = units.get_ancestor('NWBFile').trials

    order, group_inds, labels = get_group_inds_and_order(trials, trials_select, group_by, order_by)

    data = align_by_time_intervals(units, index, trials, start_label, start_label, before, after, order)
    # expanded data so that gaussian smoother uses larger window than is viewed
    expanded_data = align_by_time_intervals(units, index, trials, start_label, start_label,
                                            before + sigma_in_secs * 4,
                                            after + sigma_in_secs * 4,
                                            order)

    fig, axs = plt.subplots(2, 1)
    show_psth_raster(data, before, after, group_inds, labels, ax=axs[0])
    show_psth_smoothed(expanded_data, axs[1], before, after, group_inds,
                       sigma_in_secs=sigma_in_secs, ntt=ntt)

    axs[0].set_title('PSTH for unit {}'.format(index))
    axs[0].set_xticks([])
    axs[0].set_xlabel('')
    return fig


def compute_smoothed_firing_rate(spike_times, t, sigma_in_secs):
    """ Evaluate gaussian smoothing of spike_times at uniformly spaced array t
    Args:
      spike_times:
          A 1D numpy ndarray of spike times
      t:
          1D array, uniformly spaced, e.g. the output of np.linspace or np.arange
      sigma_in_secs:
          standard deviation of the smoothing gaussian
    Returns:
          Gaussian smoothing evaluated at array t
    """

    dt = t[1] - t[0]
    defaultlimits = (t[0], t[-1]+dt)
    bins = int((defaultlimits[1] - defaultlimits[0])/dt + 0.5)
    binned_spikes, _ = np.histogram(spike_times, bins=bins, range=defaultlimits)
    binned_spikes = binned_spikes.astype(np.float)
    sigma_in_samps = sigma_in_secs/dt
    smooth_fr = scipy.ndimage.gaussian_filter1d(binned_spikes, sigma_in_samps)/dt
    return smooth_fr


def psth(data=None, sig=0.05, T=None, err=2, t=None, num_bootstraps=10):
    """ Find peristimulus time histogram smoothed by a gaussian kernel
        The time units of the arrays in data, sig and t
        should be the same, e.g. seconds
    Args:
      data:
            A dictionary of channel names, and 1D numpy ndarray spike times,
            A numpy ndarray, where each row gives the spike times for each channel
            A 1D numpy ndarray, one list or tuple of floats that gives spike times for only one channel
      sig:  standard deviation of the smoothing gaussian. default 0.05
      T: time interval [a,b], spike times strictly outside this interval are excluded
      err: An integer, 0, 1, or 2. default 2
            0 indicates no standard error computation
            1 Poisson error
            2 Boostrap method over trials
      t: 1D array, list or tuple indicating times to evaluate psth at
      num_bootstraps: number of bootstraps. Effective only in computing error when err=2. default 10
    Returns:
      R: Rate, mean smoothed peristimulus time histogram
      t: 1D array, list or tuple indicating times psth is evaluated at
      E: standard error
    """

    # verify data argument
    try:
        if isinstance(data, dict):
            data = data.values()
        elif isinstance(data[0], (float, int)):
            data = np.array([data])
        elif isinstance(data[0], (np.ndarray, list, tuple)):
            data = np.array(data)
        else:
            raise TypeError
        if isinstance(data, np.ndarray) and len(data.shape) == 2:
            if data.shape[0] == 0 or data.shape[1] == 0:
                raise TypeError
        else:
            data = [np.array(ch_data) for ch_data in data]
            for ch_data in data:
                if len(ch_data.shape) != 1 or not isinstance(ch_data[0], (float, int)):
                    raise TypeError
    except Exception as exc:
        msg = ("psth requires spike time data as first positional argument. " +
               "Spike time data should be in the form of:\n" +
               "   a dictionary of channel names, and 1D numpy ndarray spike times\n" +
               "   a 2D numpy ndarray, where each row represents the spike times of a channel", )
        exc.args = msg
        raise exc

    # input data size
    num_channels = len(data)
    channel_lengths = [len(ch_data) for ch_data in data]
    max_channel_length = max(channel_lengths)

    if not isinstance(sig, (float, int)) or sig <= 0:
        raise TypeError("sig must be positive. Only the non-adaptive method is supported")
    if not isinstance(num_bootstraps, int) or num_bootstraps <= 0:
        raise TypeError("num_bootstraps must be a positive integer")

    # determine the interval of interest T, and mask times outside of the interval
    if T is not None:
        # expand T to avoid edge effects in rate
        T = [T[0]-4*sig, T[1]+4*sig]
        data = [np.ma.masked_outside(np.ravel(ch_data), T[0], T[1]) for ch_data in data]
    else:
        T = [np.ma.min([np.ma.min(c) for c in data]), np.ma.max([np.ma.max(c) for c in data])]

    # determine t
    if t is None:
        num_points = int(5*(T[1]-T[0])/sig)
        t = np.linspace(T[0], T[1], num_points)
        t_min, t_max = T[0], T[1]
    else:
        t = np.ravel(t)
        num_points = len(t)
        t_min, t_max = np.min(t), np.max(t)

    # masked input data size
    data_lengths = [np.ma.count(ch_data) for ch_data in data]
    num_times_total = sum(data_lengths)+1

    # Transposes data if necessary. The longer dimension is assumed to be time.
    # This is to convert the data into a form compatible with Murray's routine
    if num_channels < max_channel_length:
        num_t = num_channels  # number of trials
    else:
        num_t = max_channel_length
        # pad and transpose
        data_transposed = np.ma.zeros((max_channel_length, num_channels))
        data_transposed[...] = np.ma.masked
        for n in range(num_channels):
            data_transposed[:, n] = data[n]
        data = data_transposed
        del data_transposed
        max_channel_length = num_channels

    # warn if spikes have low density
    L = num_times_total/(num_t*(T[1]-T[0]))
    if 2*L*num_t*sig < 1 or L < 0.1:
        print('Spikes have very low density. The time units may not be the same, or the kernel width is too small')
        print('Total events: %f \nsig: %f ms \nT: %f \nevents*sig: %f\n'
              % (num_times_total, sig*1000, T, num_times_total*sig/(T[1]-T[0])))

    # fine grid in case t does not have sufficient precision for gaussian_filter1d
    num_points_extended = 6*int(5*(t_max-t_min)/sig)
    t_extended = np.linspace(t_min, t_max, num_points_extended)
    smooth_fr_index = np.rint((t-t_min)/(t_extended[1]-t_extended[0])).astype(np.int)
    # evaluate kernel density estimation at array t
    RR = np.zeros((num_t, num_points))
    for n in range(num_t):
        spike_times = data[n] if not np.ma.is_masked(data[n]) else data[n].compressed()
        smooth_fr = compute_smoothed_firing_rate(spike_times, t_extended, sig)
        RR[n, :] = smooth_fr[smooth_fr_index]

    # find rate
    R = np.mean(RR, axis=0)

    # find error
    if num_t < 4 and err == 2:
        print('Switching to Poisson errorbars as number of trials is too small for bootstrap')
        err = 1
    # std dev is sqrt(rate*(integral over kernal^2)/trials)
    # for Gaussian integral over Kernal^2 is 1/(2*sig*srqt(pi))
    if err == 0:
        E = None
    elif err == 1:
        E = np.sqrt(R/(2*num_t*sig*np.sqrt(np.pi)))
    elif err == 2:
        mean_ = [np.mean(RR[np.random.randint(0, num_t), :]) for _ in range(num_bootstraps)]
        E = np.std(mean_)
    else:
        raise TypeError("err must be 0, 1, or 2")

    return R, t, E


def show_psth_smoothed(data, ax, before, after, group_inds=None, sigma_in_secs=.05, ntt=1000):

    all_data = np.hstack(data)
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


def plot_grouped_events(data, window, group_inds=None, colors=color_wheel, ax=None, labels=None,
                        show_legend=True):
    data = np.asarray(data)
    if ax is None:
        fig, ax = plt.subplots()
    if group_inds is not None:
        handles = []
        for i, color in zip(np.unique(group_inds), colors):
            lineoffsets = np.where(group_inds == i)[0]
            event_collection = ax.eventplot(data[group_inds == i],
                                            orientation='horizontal',
                                            lineoffsets=lineoffsets,
                                            color=color)
            handles.append(event_collection[0])
        if show_legend:
            ax.legend(handles=handles, labels=list(labels),
                      bbox_to_anchor=(1.0, .6, .4, .4),
                      mode="expand", borderaxespad=0.)
    else:
        ax.eventplot(data, orientation='horizontal', color='k')

    ax.set_xlim(window)
    ax.set_ylim(-.5, len(data) - .5)
    ax.set_xlabel('time (s)')

    return ax


def show_psth_raster(data, before, after, group_inds=None, labels=None, ax=None, show_legend=True):
    ax = plot_grouped_events(data, [-before, after], group_inds, color_wheel, ax, labels,
                             show_legend=show_legend)
    ax.set_ylabel('trials')
    ax.axvline(color=[0.7, 0.7, 0.7])
    return ax


def raster_grid(units, trials, index, before, after, rows_label=None, cols_label=None, align_by='start_time'):
    if trials is None:
        raise ValueError('trials must exist (trials cannot be None)')
    if rows_label is not None:
        row_vals = np.unique(trials[rows_label][:])
    else:
        row_vals = [None]
    nrows = len(row_vals)

    if cols_label is not None:
        col_vals = np.unique(trials[cols_label][:])
    else:
        col_vals = [None]
    ncols = len(col_vals)

    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)
    big_ax = create_big_ax(fig)
    for i, row in enumerate(row_vals):
        for j, col in enumerate(col_vals):
            ax = axs[i, j]
            trials_select = np.ones((len(trials),)).astype('bool')
            if row is not None:
                trials_select &= np.array(trials[rows_label][:]) == row
            if col is not None:
                trials_select &= np.array(trials[cols_label][:]) == col
            trials_select = np.where(trials_select)[0]
            data = align_by_time_intervals(units, index, trials, align_by, align_by,
                                           before, after, list(trials_select))
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


def raster_grid_widget(units: Units):

    trials = units.get_ancestor('NWBFile').trials
    if trials is None:
        return widgets.HTML('No trials present')

    groups = infer_categorical_columns(trials)

    control_widgets = widgets.VBox(children=[])

    rows_controller = widgets.Dropdown(options=[None] + list(groups), description='rows: ',
                                       layout=Layout(width='95%'))
    cols_controller = widgets.Dropdown(options=[None] + list(groups), description='cols: ',
                                       layout=Layout(width='95%'))
    control_widgets.children = list(control_widgets.children) + [rows_controller, cols_controller]

    trial_event_controller = make_trial_event_controller(trials)
    control_widgets.children = list(control_widgets.children) + [trial_event_controller]

    unit_controller = int_controller(len(units['spike_times'].data) - 1)
    control_widgets.children = list(control_widgets.children) + [unit_controller]

    before_slider = widgets.FloatSlider(.5, min=0, max=5., description='before (s)', continuous_update=False)
    control_widgets.children = list(control_widgets.children) + [before_slider]

    after_slider = widgets.FloatSlider(2., min=0, max=5., description='after (s)', continuous_update=False)
    control_widgets.children = list(control_widgets.children) + [after_slider]

    controls = {
        'units': fixed(units),
        'trials': fixed(trials),
        'index': unit_controller.children[0],
        'after': after_slider,
        'before': before_slider,
        'align_by': trial_event_controller,
        'rows_label': rows_controller,
        'cols_label': cols_controller
    }

    out_fig = widgets.interactive_output(raster_grid, controls)
    vbox = widgets.VBox(children=[control_widgets, out_fig])

    return vbox
