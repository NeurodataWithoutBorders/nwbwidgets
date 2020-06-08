from .base import lazy_show_over_data, GroupingWidget
from .timeseries import show_timeseries_mpl
from ipywidgets import widgets
import matplotlib.pyplot as plt
from ndx_icephys_meta.icephys import SequentialRecordingsTable
from functools import partial
import numpy as np
from matplotlib.pyplot import Figure
import pandas as pd


def show_single_sequential_recording(sequential_recording, axs=None, title=None, **kwargs) -> Figure:
    """
    Show a single rep of a single stimulus sequence

    Parameters
    ----------
    sequential_recording
    axs: [matplotlib.pyplot.Axes, matplotlib.pyplot.Axes], optional
    title: str, optional
    kwargs: dict
        passed to show_timeseries_mpl

    Returns
    -------

    matplotlib.pyplot.Figure

    """

    nsweeps = len(sequential_recording)
    if axs is None:
        fig, axs = plt.subplots(2, 1, sharex=True)
    else:
        fig = axs[0].get_figure()
    for i in range(nsweeps):
        start, stop, ts = sequential_recording['recordings'].iloc[i]['responses']['response'].iloc[0]
        show_timeseries_mpl(ts, istart=start, istop=stop, ax=axs[0], zero_start=True, xlabel='', title=title, **kwargs)

        start, stop, ts = sequential_recording['recordings'].iloc[i]['stimuli']['stimulus'].iloc[0]
        show_timeseries_mpl(ts, istart=start, istop=stop, ax=axs[1], zero_start=True, **kwargs)
    return fig


def show_sequential_recordings_reps(stim_df: pd.DataFrame, **kwargs) -> Figure:
    """
    Show data from multiple repetitions of the same stimulus type

    Parameters
    ----------
    stim_df: pandas.DataFrame
    kwargs: dict
        passed to show_single_sequential_recording

    Returns
    -------
    matplotlib.pyplot.Figure

    """
    nsweeps = len(stim_df['simultaneous_recordings'])

    if 'repetition' in stim_df:
        stim_df = stim_df.sort_values('repetition')
    fig, axs = plt.subplots(2, nsweeps, sharex='col', sharey='row', figsize=[6.4 * nsweeps, 4.8])
    if nsweeps == 1:
        axs = np.array([axs]).T
    for i, (sweep, sweep_axs) in enumerate(zip(stim_df['simultaneous_recordings'], axs.T)):
        if i:
            kwargs.update(ylabel='')
        show_single_sequential_recording(sweep, axs=sweep_axs, title='Repetition {}'.format(i+1), **kwargs)
    return fig


def show_sequential_recordings(node: SequentialRecordingsTable, *args, style: GroupingWidget = widgets.Accordion, **kwargs) -> \
        GroupingWidget:
    """
    Visualize the sequential recordings table with a lazy accordion of stimulus types

    Parameters
    ----------
    node: SequentialRecordingsTable
    style: widgets.Accordion or widgets.Tabs

    Returns
    -------
    widgets.Accordion or widgets.Tabs

    """
    if 'stimulus_type' in node:
        labels, data = zip(*[(stim_label, stim_df)
                             for stim_label, stim_df in node.to_dataframe().groupby('stimulus_type')])
        func_ = show_sequential_recordings_reps
    else:
        data = node['sweeps']
        labels = None
        func_ = show_single_sweep_sequence
    func_ = partial(func_, **kwargs)
    return lazy_show_over_data(data, func_, labels=labels, style=style)
