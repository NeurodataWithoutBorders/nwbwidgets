from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import widgets
from ndx_icephys_meta.icephys import SweepSequences
from plotly.subplots import make_subplots
from pynwb.icephys import SequentialRecordingsTable

from .base import GroupingWidget, lazy_show_over_data
from .timeseries import show_indexed_timeseries_mpl


def show_single_sweep_sequence(sweep_sequence, axs=None, title=None, **kwargs) -> plt.Figure:
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
        start, stop, ts = sweep_sequence["recordings"].iloc[i]["response"].iloc[0][0]
        show_indexed_timeseries_mpl(
            ts, istart=start, istop=stop, ax=axs[0], zero_start=True, xlabel="", title=title, **kwargs
        )

        start, stop, ts = sweep_sequence["recordings"].iloc[i]["stimulus"].iloc[0][0]
        show_indexed_timeseries_mpl(ts, istart=start, istop=stop, ax=axs[1], zero_start=True, **kwargs)
    return fig


def show_sweep_sequence_reps(stim_df: pd.DataFrame, **kwargs) -> plt.Figure:
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
    nsweeps = len(stim_df["sweeps"])

    if "repetition" in stim_df:
        stim_df = stim_df.sort_values("repetition")
    fig, axs = plt.subplots(2, nsweeps, sharex="col", sharey="row", figsize=[6.4 * nsweeps, 4.8])
    if nsweeps == 1:
        axs = np.array([axs]).T
    for i, (sweep, sweep_axs) in enumerate(zip(stim_df["sweeps"], axs.T)):
        if i:
            kwargs.update(ylabel="")
        show_single_sweep_sequence(sweep, axs=sweep_axs, title="rep {}".format(i + 1), **kwargs)
    return fig


def show_sweep_sequences(
    node: SweepSequences, *args, style: GroupingWidget = widgets.Accordion, **kwargs
) -> GroupingWidget:
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
    if "stimulus_type" in node:
        labels, data = zip(
            *[(stim_label, stim_df) for stim_label, stim_df in node.to_dataframe().groupby("stimulus_type")]
        )
        func_ = show_sweep_sequence_reps
    else:
        data = node["sweeps"]
        labels = None
        func_ = show_single_sweep_sequence
    func_ = partial(func_, **kwargs)
    return lazy_show_over_data(data, func_, labels=labels, style=style)


def show_sequential_recordings(nwbfile, elec_name, sequence_id=0):
    color_wheel = px.colors.qualitative.D3

    stimulus_type = nwbfile.icephys_sequential_recordings[sequence_id]["stimulus_type"].values[0]
    curve_type = "Stimulus-Response curve"

    simultaneous_ids = nwbfile.icephys_sequential_recordings[sequence_id]["simultaneous_recordings"].values[0]
    recordings_ids = np.array([])
    for i in nwbfile.icephys_simultaneous_recordings[simultaneous_ids]["recordings"]:
        recordings_ids = np.append(recordings_ids, i)

    filtered_elec_ids = list()
    for i, row in nwbfile.intracellular_recordings["electrodes"].iterrows():
        if row[0].name == elec_name:
            filtered_elec_ids.append(i)

    filtered_ids = [int(i) for i in np.intersect1d(recordings_ids, filtered_elec_ids)]

    fig = go.FigureWidget(
        make_subplots(
            rows=2, cols=2, specs=[[{}, {"rowspan": 2}], [{}, None]], subplot_titles=("", curve_type, stimulus_type)
        )
    )

    iv_curve_x = list()
    iv_curve_y = list()
    ii = 0
    for i, row in nwbfile.intracellular_recordings.to_dataframe().loc[filtered_ids].iterrows():
        if ii == 0:
            response_unit = row.responses.response.timeseries.unit
            stimulus_unit = row.stimuli.stimulus.timeseries.unit

        response_conversion = row.responses.response.timeseries.conversion
        response_data = np.array(row.responses.response.timeseries.data[:]) * response_conversion
        response_rate = row.responses.response.timeseries.rate
        response_x = np.arange(len(response_data)) / response_rate

        if row.stimuli.stimulus.timeseries:
            stimulus_data = row.stimuli.stimulus.timeseries.data[:]
            stimulus_rate = row.stimuli.stimulus.timeseries.rate
        else:
            stimulus_data = np.zeros(len(response_data))
            stimulus_rate = response_rate
        stimulus_x = np.arange(len(stimulus_data)) / stimulus_rate

        # I-V curve
        if (max(stimulus_data) - stimulus_data[0]) > 0:
            iv_curve_x_point = max(stimulus_data)
        else:
            iv_curve_x_point = min(stimulus_data)
        iv_curve_x.append(iv_curve_x_point)

        abs_response_data = np.absolute(response_data - response_data[0])
        ind = np.argmax(abs_response_data)
        iv_curve_y.append(response_data[ind])

        fig.add_trace(
            go.Scatter(
                x=response_x,
                y=response_data,
                legendgroup=f"{int(ii)}",
                name=f"Sweep {int(ii)}",
                marker=dict(color=color_wheel[int(ii % 10)]),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=stimulus_x,
                y=stimulus_data,
                legendgroup=f"{int(ii)}",
                showlegend=False,
                marker=dict(color=color_wheel[int(ii % 10)]),
            ),
            row=2,
            col=1,
        )

        ii += 1

    fig.add_trace(
        go.Scatter(x=iv_curve_x, y=iv_curve_y, showlegend=False, mode="lines", line=dict(color="black", width=2)),
        row=1,
        col=2,
    )

    for ii in range(len(iv_curve_x)):
        fig.add_trace(
            go.Scatter(
                x=[iv_curve_x[ii]],
                y=[iv_curve_y[ii]],
                showlegend=False,
                legendgroup=f"{int(ii)}",
                marker=dict(color=color_wheel[int(ii % 10)], size=10),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        height=600, width=1200, legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(title_text="time [s]", showgrid=False, row=2, col=1)
    fig.update_xaxes(title_text=f"stimuli [{stimulus_unit}]", showgrid=False, row=1, col=2)
    fig.update_xaxes(showline=False, zeroline=True, zerolinewidth=1, zerolinecolor="black")

    fig.update_yaxes(title_text=f"response [{response_unit}]", showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text=f"stimuli [{stimulus_unit}]", showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text=f"response [{response_unit}]", showgrid=False, row=1, col=2)
    fig.update_yaxes(showline=False, zeroline=True, zerolinewidth=1, zerolinecolor="black")

    return fig


class IVCurveWidget(widgets.VBox):
    def __init__(self, sequential_recordings_table: SequentialRecordingsTable, neurodata_vis_spec=None, **kwargs):
        super().__init__()

        self.table = sequential_recordings_table

        # Electrodes
        elec_options = [(n, i) for i, n in enumerate(self.table.get_ancestor().ic_electrodes.keys())]
        self.electrode_name = elec_options[0][0]
        dropdown_elec = widgets.Dropdown(
            options=elec_options,
            value=0,
            description="Electrode:",
        )
        dropdown_elec.observe(self.update_electrode)

        # Stimuli
        self.stimuli_index = 0
        dropdown_stim = widgets.Dropdown(
            options=[(v, i) for i, v in enumerate(list(self.table.stimulus_type[:]))],
            value=0,
            description="Stimulus:",
        )
        dropdown_stim.observe(self.update_stimulus)

        self.iv_curve_controller = widgets.HBox([dropdown_elec, dropdown_stim])

        self.update_figure()

        self.children = [self.iv_curve_controller, self.fig]

    def update_electrode(self, change):
        if change["name"] == "label":
            self.electrode_name = change["new"]
            self.update_figure()

    def update_stimulus(self, change):
        if change["name"] == "value":
            self.stimuli_index = change["new"]
            self.update_figure()

    def update_figure(self):
        self.fig = show_sequential_recordings(
            nwbfile=self.table.get_ancestor(), elec_name=self.electrode_name, sequence_id=self.stimuli_index
        )
        self.children = [self.iv_curve_controller, self.fig]
