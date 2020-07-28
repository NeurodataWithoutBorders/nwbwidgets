from nwbwidgets.utils.timeseries import get_timeseries_maxt, get_timeseries_mint
from nwbwidgets.controllers import StartAndDurationController
from nwbwidgets.timeseries import SingleTracePlotlyWidget
from nwbwidgets.image import ImageSeriesWidget
import plotly.graph_objects as go
from ipywidgets import widgets, Layout
from tifffile import imread
from pathlib import Path, PureWindowsPath
import numpy as np


class AllenDashboard(widgets.VBox):
    def __init__(self, nwb):
        super().__init__()
        self.nwb = nwb
        self.show_spikes = False

        self.btn_spike_times = widgets.Button(description='Show spike times', button_style='')
        self.btn_spike_times.on_click(self.spikes_viewer)

        # Start time and duration controller
        self.tmin = get_timeseries_mint(nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'])
        self.tmax = get_timeseries_maxt(nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'])
        self.time_window_controller = StartAndDurationController(
            tmin=self.tmin,
            tmax=self.tmax,
            start=0,
            duration=5,
        )

        # Electrophys single trace
        self.electrical = SingleTracePlotlyWidget(
            timeseries=nwb.processing['ecephys'].data_interfaces['filtered_membrane_voltage'],
            foreign_time_window_controller=self.time_window_controller,
        )
        self.electrical.out_fig.update_layout(
            title=None,
            showlegend=False,
            xaxis_title=None,
            yaxis_title='Volts',
            width=840,
            height=230,
            margin=dict(l=60, r=200, t=8, b=20),
            # yaxis={"position": 0, "anchor": "free"},
            yaxis={"range": [min(self.electrical.out_fig.data[0].y), max(self.electrical.out_fig.data[0].y)],
                   "autorange": False},
            xaxis={"showticklabels": False, "ticks": "",
                   "range": [min(self.electrical.out_fig.data[0].x), max(self.electrical.out_fig.data[0].x)],
                   "autorange": False}
        )
        # Fluorescence single trace
        self.fluorescence = SingleTracePlotlyWidget(
            timeseries=nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'],
            foreign_time_window_controller=self.time_window_controller,
        )
        self.fluorescence.out_fig.update_layout(
            title=None,
            showlegend=False,
            width=840,
            height=230,
            margin=dict(l=60, r=200, t=8, b=20),
            yaxis_title='dF/F',
            yaxis={"range": [min(self.fluorescence.out_fig.data[0].y), max(self.fluorescence.out_fig.data[0].y)],
                   "autorange": False},
            xaxis={"range": [min(self.fluorescence.out_fig.data[0].x), max(self.fluorescence.out_fig.data[0].x)],
                   "autorange": False, "constrain": "domain", "anchor": "free"}
        )
        # Two photon imaging
        self.photon_series = ImageSeriesWidget(
            imageseries=nwb.acquisition['raw_ophys'],
            foreign_time_window_controller=self.time_window_controller,
        )
        self.photon_series.out_fig.update_layout(
            showlegend=False,
            margin=dict(l=30, r=5, t=65, b=65),
        )

        # Frame controller
        self.frame_controller = widgets.FloatSlider(
            value=0,
            step=1 / self.nwb.acquisition['raw_ophys'].rate,
            min=self.time_window_controller.value[0],
            max=self.time_window_controller.value[1],
            description='Frame: ',
            style={'description_width': '55px'},
            continuous_update=False,
            readout=False,
            orientation='horizontal',
            layout=Layout(width='645px'),
        )

        # Add line traces marking Image frame point
        self.frame_point = go.Scatter(x=[0, 0], y=[-1000, 1000])
        self.electrical.out_fig.add_trace(self.frame_point)
        self.fluorescence.out_fig.add_trace(self.frame_point)

        # Updates frame point
        self.frame_controller.observe(self.update_frame_point)

        # Updates list of valid spike times at each change in time range
        self.time_window_controller.observe(self.updated_time_range)

        # Layout
        hbox_header = widgets.HBox([self.btn_spike_times, self.time_window_controller])
        vbox_widgets = widgets.VBox([self.electrical, self.fluorescence])
        hbox_widgets = widgets.HBox([vbox_widgets, self.photon_series])

        self.children = [hbox_header, self.frame_controller, hbox_widgets]

        self.update_spike_traces()

    def update_frame_point(self, change):
        """Updates Image frame and frame point relative position on temporal traces"""
        if isinstance(change['new'], float):
            self.electrical.out_fig.data[1].x = [change['new'], change['new']]
            self.fluorescence.out_fig.data[1].x = [change['new'], change['new']]

            frame_number = int(change['new'] * self.nwb.acquisition['raw_ophys'].rate)
            file_path = self.nwb.acquisition['raw_ophys'].external_file[0]
            if "\\" in file_path:
                win_path = PureWindowsPath(file_path)
                path_ext_file = Path(win_path)
            else:
                path_ext_file = Path(file_path)
            image = imread(path_ext_file, key=frame_number)
            self.photon_series.out_fig.data[0].z = image

    def updated_time_range(self, change=None):
        """Operations to run whenever time range gets updated"""
        self.update_spike_traces()
        self.show_spikes = False

        # Update frame slider
        if self.time_window_controller.value[1] < self.frame_controller.min:
            self.frame_controller.min = self.time_window_controller.value[0]
            self.frame_controller.max = self.time_window_controller.value[1]
        else:
            self.frame_controller.max = self.time_window_controller.value[1]
            self.frame_controller.min = self.time_window_controller.value[0]
        xpoint = round(np.mean(self.time_window_controller.value))
        self.frame_controller.value = xpoint
        self.electrical.out_fig.data[1].x = [xpoint, xpoint]
        self.fluorescence.out_fig.data[1].x = [xpoint, xpoint]

        # Reset spike times view
        self.btn_spike_times.description = 'Show spike times'
        self.fluorescence.out_fig.data = [
            self.fluorescence.out_fig.data[0],
            self.fluorescence.out_fig.data[1]
        ]
        self.electrical.out_fig.data = [
            self.electrical.out_fig.data[0],
            self.electrical.out_fig.data[1]
        ]

    def spikes_viewer(self, b=None):
        self.show_spikes = not self.show_spikes
        if self.show_spikes:
            self.btn_spike_times.description = 'Hide spike times'
            for spike_trace in self.spike_traces:
                self.fluorescence.out_fig.add_trace(spike_trace)
        else:
            self.btn_spike_times.description = 'Show spike times'
            self.fluorescence.out_fig.data = [
                self.fluorescence.out_fig.data[0],
                self.fluorescence.out_fig.data[1]
            ]

    def update_spike_traces(self):
        """Updates list of go.Scatter objects at spike times"""
        self.spike_traces = []
        t_start = self.time_window_controller.value[0]
        t_end = self.time_window_controller.value[1]
        all_spikes = self.nwb.units['spike_times'][0]
        mask = (all_spikes > t_start) & (all_spikes < t_end)
        selected_spikes = all_spikes[mask]
        # Makes a go.Scatter object for each spike in chosen interval
        for spkt in selected_spikes:
            self.spike_traces.append(go.Scatter(
                x=[spkt, spkt],
                y=[-1000, 1000],
                line={"color": "gray", "width": .5}
            ))
