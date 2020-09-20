from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import widgets
from matplotlib.pyplot import Figure
from ndx_icephys_meta.icephys import SweepSequences

from .base import lazy_show_over_data, GroupingWidget
from .timeseries import show_indexed_timeseries_mpl


def show_single_sweep_sequence(sweep_sequence, axs=None, title=None, **kwargs) -> Figure:
    """
    Show a single rep of a single stimulus sequence

    Parameters
    ----------
    sweep_sequence
    axs: [matplotlib.pyplot.Axes, matplotlib.pyplot.Axes], optional
    title: str, optional
    kwargs: dict
        passed to show_indexed_timeseries_mpl

    Returns
    -------

    matplotlib.pyplot.Figure

    """

    nsweeps = len(sweep_sequence)
    if axs is None:
        fig, axs = plt.subplots(2, 1, sharex=True)
    else:
        fig = axs[0].get_figure()
    for i in range(nsweeps):
        start, stop, ts = sweep_sequence['recordings'].iloc[i]['response'].iloc[0][0]
        show_indexed_timeseries_mpl(ts, istart=start, istop=stop, ax=axs[0], zero_start=True, xlabel='', title=title,
                                    **kwargs)

        start, stop, ts = sweep_sequence['recordings'].iloc[i]['stimulus'].iloc[0][0]
        show_indexed_timeseries_mpl(ts, istart=start, istop=stop, ax=axs[1], zero_start=True, **kwargs)
    return fig


def show_sweep_sequence_reps(stim_df: pd.DataFrame, **kwargs) -> Figure:
    """
    Show data from multiple reps of the same stimulus type

    Parameters
    ----------
    stim_df: pandas.DataFrame
    kwargs: dict
        passed to show_single_sweep_sequence

    Returns
    -------
    matplotlib.pyplot.Figure

    """
    nsweeps = len(stim_df['sweeps'])

    if 'repetition' in stim_df:
        stim_df = stim_df.sort_values('repetition')
    fig, axs = plt.subplots(2, nsweeps, sharex='col', sharey='row', figsize=[6.4 * nsweeps, 4.8])
    if nsweeps == 1:
        axs = np.array([axs]).T
    for i, (sweep, sweep_axs) in enumerate(zip(stim_df['sweeps'], axs.T)):
        if i:
            kwargs.update(ylabel='')
        show_single_sweep_sequence(sweep, axs=sweep_axs, title='rep {}'.format(i + 1), **kwargs)
    return fig


def show_sweep_sequences(node: SweepSequences, *args, style: GroupingWidget = widgets.Accordion, **kwargs) -> \
        GroupingWidget:
    """
    Visualize the sweep sequences table with a lazy accordion of sweep sequence repetitions

    Parameters
    ----------
    node: SweepSequences
    style: widgets.Accordion or widgets.Tabs

    Returns
    -------
    widgets.Accordion or widgets.Tabs

    """
    if 'stimulus_type' in node:
        labels, data = zip(*[(stim_label, stim_df)
                             for stim_label, stim_df in node.to_dataframe().groupby('stimulus_type')])
        func_ = show_sweep_sequence_reps
    else:
        data = node['sweeps']
        labels = None
        func_ = show_single_sweep_sequence
    func_ = partial(func_, **kwargs)
    return lazy_show_over_data(data, func_, labels=labels, style=style)
