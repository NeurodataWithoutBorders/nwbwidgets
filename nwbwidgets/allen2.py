from ipywidgets import widgets
# from nwbwidgets.utils.timeseries import get_timeseries_maxt, get_timeseries_mint
from .controllers import StartAndDurationController
# from .ophys import TwoPhotonSeriesWidget
from .timeseries import SingleTracePlotlyWidget
from .image import ImageSeriesWidget


class AllenDashboard(widgets.VBox):
    def __init__(self, nwb):
        super().__init__()

        # self.tmin = get_timeseries_mint(time_series)
        # self.tmax = get_timeseries_maxt(time_series)
        self.lines_select = False

        self.btn_lines = widgets.Button(description='Show spike times', button_style='')
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
            width=600,
            height=230,
            margin=dict(l=0, r=8, t=8, b=8),
            xaxis={"showticklabels": False, "ticks": ""},
            # yaxis={"position": 0, "anchor": "free"}
        )
        # Fluorescence single trace
        self.fluorescence = SingleTracePlotlyWidget(
            timeseries=nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'],
            foreign_time_window_controller=self.time_window_controller,
        )
        self.fluorescence.out_fig.update_layout(
            title=None,
            width=600,
            height=230,
            margin=dict(l=65, r=8, t=8, b=8),
        )
        # Two photon imaging
        self.photon_series = ImageSeriesWidget(
            imageseries=nwb.acquisition['raw_ophys'],
            foreign_time_window_controller=self.time_window_controller,
        )
        self.photon_series.out_fig.update_layout(
            margin=dict(l=30, r=5, t=35, b=35),
        )

        hbox_header = widgets.HBox([self.btn_lines, self.time_window_controller])
        vbox_widgets = widgets.VBox([self.electrical, self.fluorescence])
        hbox_widgets = widgets.HBox([vbox_widgets, self.photon_series])

        self.children = [hbox_header, hbox_widgets]

    def btn_lines_dealer(self, b=0):
        self.lines_select = not self.lines_select
        if self.lines_select:
            self.btn_lines.description = 'Show spike times'
        else:
            self.btn_lines.description = 'Hide spike times'
