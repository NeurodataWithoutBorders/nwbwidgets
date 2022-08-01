import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from ipywidgets import widgets, ValueWidget

from pynwb import TimeSeries
from pynwb.base import DynamicTable
from pynwb.ecephys import SpikeEventSeries, ElectricalSeries

from .base import fig2widget, lazy_tabs, render_dataframe
from .timeseries import BaseGroupedTraceWidget
from .brains import HumanElectrodesPlotlyWidget
from .utils.dependencies import safe_import, check_widget_dependencies

ccfwidget = safe_import('ccfwidget')


def show_spectrogram(nwbobj: TimeSeries, channel=0, **kwargs):
    fig, ax = plt.subplots()
    f, t, Zxx = stft(nwbobj.data[:, channel], nwbobj.rate, nperseg=2 * 17)
    ax.imshow(
        np.log(np.abs(Zxx)),
        aspect="auto",
        extent=[0, max(t), 0, max(f)],
        origin="lower",
    )
    ax.set_ylim(0, max(f))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    fig.show()


class ElectrodeGroupsWidget(ValueWidget, widgets.HBox):
    def __init__(self, nwbobj: DynamicTable, **kwargs):
        super().__init__()
        group_names = nwbobj.group_name[:]
        ugroups, group_pos, counts = np.unique(
            group_names, return_inverse=True, return_counts=True
        )
        elec_pos = np.hstack(np.arange(count).tolist() for count in counts)

        hovertext = []
        df = nwbobj.to_dataframe()
        for i, row in df.iterrows():
            hovertext.append("")
            for key, val in list(row.to_dict().items()):
                if key == "group":
                    continue
                hovertext[-1] += "{}: {}<br>".format(key, val)

        self.fig = go.FigureWidget()
        self.fig.add_trace(
            go.Scatter(
                x=elec_pos,
                y=nwbobj.group_name[:],
                mode="markers",
                marker=dict(
                    color=np.array(DEFAULT_PLOTLY_COLORS)[
                        group_pos % len(DEFAULT_PLOTLY_COLORS)
                    ],
                    size=15,
                ),
                hovertext=hovertext,
                hoverinfo="text",
            )
        )

        self.fig.update_layout(
            width=400,
            height=700,
            xaxis_title="group electrode number",
            showlegend=False,
            margin=dict(t=10),
        )

        self.value = list(range(len(group_names)))

        def selection_fn(trace, points, selector):
            self.value = points.point_inds

        self.fig.data[0].on_selection(selection_fn)

        self.children = [self.fig]


def show_electrodes(electrodes_table):
    in_dict = dict(table=render_dataframe)
    if np.isnan(electrodes_table.x[0]):  # position is not defined
        in_dict.update(electrode_groups=ElectrodeGroupsWidget)
    else:
        subject = electrodes_table.get_ancestor("NWBFile").subject
        if subject is not None:
            species = subject.species
            if species in ("mouse", "Mus musculus"):
                in_dict.update(CCF=show_ccf)
            elif species in ("human", "Homo sapiens"):
                in_dict.update(render=HumanElectrodesPlotlyWidget)

    return lazy_tabs(in_dict, electrodes_table)


@check_widget_dependencies({'ccfwidget' : ccfwidget})
def show_ccf(electrodes_table=None, **kwargs):
    from ccfwidget import CCFWidget

    input_kwargs = {}
    if electrodes_table is not None:
        df = electrodes_table.to_dataframe()
        markers = [
            idf[["x", "y", "z"]].to_numpy() for _, idf in df.groupby("group_name")
        ]
        input_kwargs.update(markers=markers)

    input_kwargs.update(kwargs)
    return CCFWidget(**input_kwargs)


def show_spike_event_series(ses: SpikeEventSeries, **kwargs):
    def control_plot(spk_ind):
        fig, ax = plt.subplots(figsize=(9, 5))
        data = ses.data[spk_ind]
        if nChannels > 1:
            for ch in range(nChannels):
                ax.plot(data[:, ch], color="#d9d9d9")
        else:
            ax.plot(data[:], color="#d9d9d9")
        ax.plot(np.mean(data, axis=1), color="k")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        fig.show()
        return fig2widget(fig)

    if len(ses.data.shape) == 3:
        nChannels = ses.data.shape[2]
    else:
        nChannels = ses.data.shape[1]
    nSpikes = ses.data.shape[0]

    # Controls
    field_lay = widgets.Layout(
        max_height="40px", max_width="100px", min_height="30px", min_width="70px"
    )
    spk_ind = widgets.BoundedIntText(value=0, min=0, max=nSpikes - 1, layout=field_lay)
    controls = {"spk_ind": spk_ind}
    out_fig = widgets.interactive_output(control_plot, controls)

    # Assemble layout box
    lbl_spk = widgets.Label("Spike ID:", layout=field_lay)
    lbl_nspks0 = widgets.Label("N° spikes:", layout=field_lay)
    lbl_nspks1 = widgets.Label(str(nSpikes), layout=field_lay)
    lbl_nch0 = widgets.Label("N° channels:", layout=field_lay)
    lbl_nch1 = widgets.Label(str(nChannels), layout=field_lay)
    hbox0 = widgets.HBox(children=[lbl_spk, spk_ind])
    vbox0 = widgets.VBox(
        children=[
            widgets.HBox(children=[lbl_nspks0, lbl_nspks1]),
            widgets.HBox(children=[lbl_nch0, lbl_nch1]),
            hbox0,
        ]
    )
    hbox1 = widgets.HBox(children=[vbox0, out_fig])

    return hbox1


class ElectricalSeriesWidget(BaseGroupedTraceWidget):
    def __init__(
        self,
        electrical_series: ElectricalSeries,
        neurodata_vis_spec=None,
        foreign_time_window_controller=None,
        foreign_group_and_sort_controller=None,
        dynamic_table_region_name="electrodes",
        **kwargs
    ):
        if foreign_group_and_sort_controller is not None:
            table = None
        else:
            table = dynamic_table_region_name
        super().__init__(
            electrical_series,
            table,
            foreign_time_window_controller=foreign_time_window_controller,
            foreign_group_and_sort_controller=foreign_group_and_sort_controller,
            **kwargs
        )
