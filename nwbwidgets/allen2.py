from ipywidgets import widgets
# from nwbwidgets.utils.timeseries import get_timeseries_maxt, get_timeseries_mint
from .controllers import StartAndDurationController
import plotly.graph_objects as go
from .timeseries import SingleTracePlotlyWidget
from .image import ImageSeriesWidget


class AllenDashboard(widgets.VBox):
    def __init__(self, nwb):
        super().__init__()
        self.nwb = nwb

        # self.tmin = get_timeseries_mint(time_series)
        # self.tmax = get_timeseries_maxt(time_series)
        self.show_spikes = False
        self.btn_spike_times = widgets.Button(description='Show spike times', button_style='')
        self.btn_spike_times.on_click(self.spikes_viewer)

        # Start time and duration controller
        self.time_window_controller = StartAndDurationController(
            tmin=0,
            tmax=120,
            start=0,
            duration=5
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
            width=600,
            height=230,
            margin=dict(l=0, r=8, t=8, b=8),
            # yaxis={"position": 0, "anchor": "free"},
            yaxis={"range": [min(self.electrical.out_fig.data[0].y), max(self.electrical.out_fig.data[0].y)],
                   "autorange": False},
            xaxis={"showticklabels": False, "ticks": ""}
        )
        # Fluorescence single trace
        self.fluorescence = SingleTracePlotlyWidget(
            timeseries=nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'],
            foreign_time_window_controller=self.time_window_controller,
        )
        self.fluorescence.out_fig.update_layout(
            title=None,
            showlegend=False,
            width=600,
            height=230,
            margin=dict(l=65, r=8, t=8, b=8),
            yaxis={"range": [min(self.fluorescence.out_fig.data[0].y), max(self.fluorescence.out_fig.data[0].y)],
                   "autorange": False},
            # xaxis={"autorange": False}
        )
        # Two photon imaging
        self.photon_series = ImageSeriesWidget(
            imageseries=nwb.acquisition['raw_ophys'],
            foreign_time_window_controller=self.time_window_controller,
        )
        self.photon_series.out_fig.update_layout(
            showlegend=False,
            margin=dict(l=30, r=5, t=35, b=35),
        )

        # Add line traces marking Image frame point
        self.frame_point = go.Scatter(x=[2, 2], y=[-1000, 1000])
        # self.electrical.out_fig.add_trace(self.frame_point)
        # self.fluorescence.out_fig.add_trace(self.frame_point)

        hbox_header = widgets.HBox([self.btn_spike_times, self.time_window_controller])
        vbox_widgets = widgets.VBox([self.electrical, self.fluorescence])
        hbox_widgets = widgets.HBox([vbox_widgets, self.photon_series])

        self.children = [hbox_header, hbox_widgets]

        # Updates list of valid spike times at each change in time range
        self.time_window_controller.observe(self.updated_time_range)

        self.update_spike_traces()

    def updated_time_range(self, change=None):
        """Operations to run whenever time range gets updated"""
        self.update_spike_traces()
        self.show_spikes = False
        self.btn_spike_times.description = 'Show spike times'
        self.fluorescence.out_fig.data = [self.fluorescence.out_fig.data[0]]
        self.electrical.out_fig.data = [self.electrical.out_fig.data[0]]

    def spikes_viewer(self, b=None):
        self.show_spikes = not self.show_spikes
        if self.show_spikes:
            self.btn_spike_times.description = 'Hide spike times'
            for spike_trace in self.spike_traces:
                self.fluorescence.out_fig.add_trace(spike_trace)
                # self.electrical.out_fig.add_trace(spike_trace)
        else:
            self.btn_spike_times.description = 'Show spike times'
            self.fluorescence.out_fig.data = [self.fluorescence.out_fig.data[0]]
            # self.electrical.out_fig.data = [self.electrical.out_fig.data[0]]

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
