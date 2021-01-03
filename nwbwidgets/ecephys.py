import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from ipywidgets import widgets, ValueWidget
from pynwb.ecephys import LFP, SpikeEventSeries, ElectricalSeries
from scipy.signal import stft
import pynwb

from .base import fig2widget, nwb2widget, lazy_tabs, render_dataframe
from .timeseries import BaseGroupedTraceWidget


def show_lfp(ndobj: LFP, neurodata_vis_spec: dict):
    lfp = list(ndobj.electrical_series.values())[0]
    return nwb2widget(lfp, neurodata_vis_spec)


def show_spectrogram(nwbobj: pynwb.TimeSeries, channel=0, **kwargs):
    fig, ax = plt.subplots()
    f, t, Zxx = stft(nwbobj.data[:, channel], nwbobj.rate, nperseg=2 * 17)
    ax.imshow(
        np.log(np.abs(Zxx)),
        aspect='auto',
        extent=[0, max(t), 0, max(f)],
        origin='lower'
    )
    ax.set_ylim(0, max(f))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequency (Hz)')
    fig.show()


class ElectrodeGroupsWidget(ValueWidget, widgets.HBox):

    def __init__(self, nwbobj: pynwb.base.DynamicTable, **kwargs):
        super().__init__()
        group_names = nwbobj.group_name[:]
        ugroups, group_pos, counts = np.unique(group_names, return_inverse=True, return_counts=True)
        elec_pos = np.hstack(np.arange(count) for count in counts)

        hovertext = []
        df = nwbobj.to_dataframe()
        for i, row in df.iterrows():
            hovertext.append('')
            for key, val in list(row.to_dict().items()):
                if key == 'group':
                    continue
                hovertext[-1] += '{}: {}<br>'.format(key, val)

        self.fig = go.FigureWidget()
        self.fig.add_trace(
            go.Scatter(
                x=elec_pos,
                y=nwbobj.group_name[:],
                mode='markers',
                marker=dict(
                    color=np.array(DEFAULT_PLOTLY_COLORS)[group_pos % len(DEFAULT_PLOTLY_COLORS)],
                    size=15
                ),
                hovertext=hovertext,
                hoverinfo='text'
            )
        )

        self.fig.update_layout(
            width=400,
            height=700,
            xaxis_title='group electrode number',
            showlegend=False,
            margin=dict(t=10)
        )

        self.value = list(range(len(group_names)))

        def selection_fn(trace, points, selector):
            self.value = points.point_inds
            print(self.value)

        self.fig.data[0].on_selection(selection_fn)

        self.children = [self.fig]


class HumanElectrodesPlotlyWidget(widgets.VBox):

    def __init__(self, electrodes: pynwb.base.DynamicTable, **kwargs):

        super().__init__()

        slider_kwargs = dict(
            value=1.,
            min=0.,
            max=1.,
            style={'description_width': 'initial'}
        )

        left_opacity_slider = widgets.FloatSlider(
            description='left hemi opacity',
            **slider_kwargs
        )

        right_opacity_slider = widgets.FloatSlider(
            description='right hemi opacity',
            **slider_kwargs
        )

        left_opacity_slider.observe(self.observe_left_opacity)
        right_opacity_slider.observe(self.observe_right_opacity)

        self.fig = go.FigureWidget()
        self.plot_human_brain()
        self.show_electrodes(electrodes)

        self.children = [
            self.fig,
            widgets.HBox([
                left_opacity_slider, right_opacity_slider
            ])

        ]

    def show_electrodes(self, electrodes: pynwb.base.DynamicTable):

        x = electrodes.x[:]
        y = electrodes.y[:]
        z = electrodes.z[:]
        group_names = electrodes.group_name[:]
        ugroups, group_inv = np.unique(group_names, return_inverse=True)

        with self.fig.batch_update():
            for i, (group, c) in enumerate(zip(ugroups, DEFAULT_PLOTLY_COLORS)):
                selx, sely, selz = x[group_inv == i], y[group_inv == i], z[group_inv == i]

                trace_kwargs = dict()
                if group == b'GRID':
                    trace_kwargs.update(mode='markers')

                self.fig.add_trace(
                    go.Scatter3d(x=selx, y=sely, z=selz, name=group,
                                 marker=dict(color=c))
                )

    def plot_human_brain(self, left_opacity=1., right_opacity=1.):

        from nilearn import datasets, surface

        mesh = datasets.fetch_surf_fsaverage('fsaverage5')

        def create_mesh(name, **kwargs):
            vertices, triangles = surface.load_surf_mesh(mesh[name])
            x, y, z = vertices.T
            i, j, k = triangles.T

            return go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                **kwargs
            )

        kwargs = dict(
            color='lightgray',
            lighting=dict(
                specular=1,
                ambient=.9,
                roughness=0.9,
                diffuse=0.9
            ),
            hoverinfo='skip',
        )

        self.fig.add_trace(create_mesh('pial_left', opacity=left_opacity, **kwargs))
        self.fig.add_trace(create_mesh('pial_right', opacity=right_opacity, **kwargs))

        self.fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
            height=500,
            margin=dict(t=20, b=0)
        )

    def observe_left_opacity(self, change):
        if 'new' in change and isinstance(change['new'], float):
            self.fig.data[0].opacity = change['new']

    def observe_right_opacity(self, change):
        if 'new' in change and isinstance(change['new'], float):
            self.fig.data[1].opacity = change['new']


def show_electrodes(electrodes_table):
    in_dict = dict(table=render_dataframe)
    if np.isnan(electrodes_table.x[0]):  # position is not defined
        in_dict.update(electrode_groups=ElectrodeGroupsWidget)
    else:
        subject = electrodes_table.get_ancestor('NWBFile').subject
        if subject is not None:
            species = subject.species
            if species in ('mouse', 'Mus musculus'):
                in_dict.update(CCF=show_ccf)
            elif species in ('human', 'Homo sapiens'):
                in_dict.update(render=HumanElectrodesPlotlyWidget)

    return lazy_tabs(in_dict, electrodes_table)


def show_ccf(electrodes_table=None, **kwargs):
    from ccfwidget import CCFWidget
    input_kwargs = {}
    if electrodes_table is not None:
        df = electrodes_table.to_dataframe()
        markers = [idf[['x', 'y', 'z']].to_numpy()
                   for _, idf in df.groupby('group_name')]
        input_kwargs.update(markers=markers)

    input_kwargs.update(kwargs)
    return CCFWidget(**input_kwargs)


def show_spike_event_series(ses: SpikeEventSeries, **kwargs):
    def control_plot(spk_ind):
        fig, ax = plt.subplots(figsize=(9, 5))
        data = ses.data[spk_ind]
        if nChannels > 1:
            for ch in range(nChannels):
                ax.plot(data[:, ch], color='#d9d9d9')
        else:
            ax.plot(data[:], color='#d9d9d9')
        ax.plot(np.mean(data, axis=1), color='k')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        fig.show()
        return fig2widget(fig)

    if len(ses.data.shape) == 3:
        nChannels = ses.data.shape[2]
    else:
        nChannels = ses.data.shape[1]
    nSpikes = ses.data.shape[0]

    # Controls
    field_lay = widgets.Layout(max_height='40px', max_width='100px',
                               min_height='30px', min_width='70px')
    spk_ind = widgets.BoundedIntText(value=0, min=0, max=nSpikes - 1,
                                     layout=field_lay)
    controls = {'spk_ind': spk_ind}
    out_fig = widgets.interactive_output(control_plot, controls)

    # Assemble layout box
    lbl_spk = widgets.Label('Spike ID:', layout=field_lay)
    lbl_nspks0 = widgets.Label('N° spikes:', layout=field_lay)
    lbl_nspks1 = widgets.Label(str(nSpikes), layout=field_lay)
    lbl_nch0 = widgets.Label('N° channels:', layout=field_lay)
    lbl_nch1 = widgets.Label(str(nChannels), layout=field_lay)
    hbox0 = widgets.HBox(children=[lbl_spk, spk_ind])
    vbox0 = widgets.VBox(children=[
        widgets.HBox(children=[lbl_nspks0, lbl_nspks1]),
        widgets.HBox(children=[lbl_nch0, lbl_nch1]),
        hbox0
    ])
    hbox1 = widgets.HBox(children=[vbox0, out_fig])

    return hbox1


class ElectricalSeriesWidget(BaseGroupedTraceWidget):
    def __init__(self, electrical_series: ElectricalSeries, neurodata_vis_spec=None,
                 foreign_time_window_controller=None, foreign_group_and_sort_controller=None,
                 dynamic_table_region_name='electrodes', **kwargs):
        if foreign_group_and_sort_controller is not None:
            table = None
        else:
            table = dynamic_table_region_name
        super().__init__(electrical_series, table,
                         foreign_time_window_controller=foreign_time_window_controller,
                         foreign_group_and_sort_controller=foreign_group_and_sort_controller,
                         **kwargs)
