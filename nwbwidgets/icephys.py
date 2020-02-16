from .base import lazy_show_over_data, GroupingWidget, df2accordion
from .timeseries import show_timeseries_mpl
from ipywidgets import widgets
import matplotlib.pyplot as plt
from ndx_icephys_meta.icephys import SweepSequences, Conditions
from functools import partial
import numpy as np
from matplotlib.pyplot import Figure
import pandas as pd
import pynwb


def show_sweep_sequence(df, axs=None, title=None, **kwargs) -> Figure:
    """
    Show a single rep of a single stimulus sequence

    Parameters
    ----------
    df
    axs: [matplotlib.pyplot.Axes, matplotlib.pyplot.Axes], optional
    title: str, optional
    kwargs: dict
        passed to show_timeseries_mpl

    Returns
    -------

    matplotlib.pyplot.Figure

    """

    if axs is None:
        fig, axs = plt.subplots(2, 1, sharex=True)
    else:
        fig = axs[0].get_figure()
    for start, stop, ts in df['response']:
        show_timeseries_mpl(ts, istart=start, istop=stop, ax=axs[0], zero_start=True, xlabel='', title=title, **kwargs)

    for start, stop, ts in df['stimulus']:
        show_timeseries_mpl(ts, istart=start, istop=stop, ax=axs[1], zero_start=True, **kwargs)

    return fig


def show_sweep_sequence_reps(df: pd.DataFrame, **kwargs) -> Figure:
    """
    Show data from multiple reps of the same stimulus type

    Parameters
    ----------
    df: pandas.DataFrame
    kwargs: dict
        passed to show_single_sweep_sequence

    Returns
    -------
    matplotlib.pyplot.Figure

    """
    nruns = df['runs_id'].nunique()
    fig, axs = plt.subplots(2, nruns, sharex='col', sharey='row', figsize=[6.4 * nruns, 4.8])
    if nruns == 1:
        axs = np.array([axs]).T
    for i, ((run, idf), iaxs) in enumerate(zip(df.groupby('runs_id'), axs.T)):
        if i:
            kwargs.update(ylabel='')
        print(iaxs)
        show_sweep_sequence(idf, axs=iaxs, title='rep {}'.format(run), **kwargs)
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
        func_ = show_sweep_sequence
    func_ = partial(func_, **kwargs)
    return lazy_show_over_data(data, func_, labels=labels, style=style)


def get_data_selector(df):
    selectors = []
    decisions = []
    for x in df.columns:
        if x not in ('stimulus', 'response', 'sweep_sequences_repetition', 'id', 'sweep_sequences_stimulus_type'):
            col_data = df[x]
            if isinstance(col_data[0], pynwb.NWBContainer):
                vals = [x.name for x in col_data[:]]
            else:
                vals = [x for x in col_data[:]]
            unique_vals = np.unique(vals)
            if len(unique_vals) > 1:
                decisions.append(x)
                selectors.append(
                    widgets.Dropdown(options=[''] + list(unique_vals),
                                     description=x, layout={'width': '250px'},
                                     style={'description_width': '150px'}))
    selectors = widgets.VBox(children=selectors)
    return selectors, decisions


def conditions_widget(conditions: Conditions, neurodata_vis_spec=None, **kwargs):
    df = conditions.to_denormalized_dataframe(flat_column_index=True)
    selectors, decisions = get_data_selector(df)

    controls = {decision: widget for decision, widget in zip(decisions, selectors.children)}

    out_widget = widgets.interactive_output(partial(show_conditions, df=df), controls)

    return widgets.HBox(children=[selectors, out_widget])


def show_conditions(df, **kwargs):

    for choice, decision in kwargs.items():
        if choice != '':
            df = df[df[choice] == decision]

    return df2accordion(df, by=['conditions', 'electrode', 'stimulus_type'],
                        func=show_sweep_sequence_reps)


