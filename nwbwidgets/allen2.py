from ipywidgets import widgets
from pynwb import TimeSeries
# from nwbwidgets.utils.timeseries import get_timeseries_maxt, get_timeseries_mint
from .controllers import StartAndDurationController
from .ophys import RoiResponseSeriesWidget, TwoPhotonSeriesWidget
from .timeseries import SingleTracePlotlyWidget


class AllenDashboard(widgets.VBox):
    def __init__(self, nwb):
        super().__init__()

        # self.tmin = get_timeseries_mint(time_series)
        # self.tmax = get_timeseries_maxt(time_series)
        self.lines_select = False

        self.btn_lines = widgets.Button(description='Enable spike times', button_style='')
        self.btn_lines.on_click(self.btn_lines_dealer)

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
            xaxis_title=None,
            width=500,
            height=300,
            margin=dict(l=8, r=8, t=8, b=8),
        )
        # Fluorescence single trace
        self.fluorescence = SingleTracePlotlyWidget(
            timeseries=nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'],
            foreign_time_window_controller=self.time_window_controller,
        )
        self.fluorescence.out_fig.update_layout(
            title=None,
            width=500,
            height=300,
            margin=dict(l=8, r=8, t=8, b=8),
        )
        # Two photon imaging
        self.photon_series = TwoPhotonSeriesWidget(
            indexed_timeseries=nwb.acquisition['raw_ophys'],
            neurodata_vis_spec=None
        )

        hbox_header = widgets.HBox([self.btn_lines, self.time_window_controller])
        vbox_widgets = widgets.VBox([self.electrical, self.fluorescence])
        hbox_widgets = widgets.HBox([vbox_widgets, self.photon_series])

        self.children = [hbox_header, hbox_widgets]

    def btn_lines_dealer(self, b=0):
        self.lines_select = not self.lines_select
        if 'disable' in self.btn_lines.description.lower():
            self.btn_lines.description = 'Show spike times'
        else:
            self.btn_lines.description = 'Hide spike times'
