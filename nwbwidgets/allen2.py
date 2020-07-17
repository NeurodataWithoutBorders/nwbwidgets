from ipywidgets import widgets
from pynwb import TimeSeries
#from nwbwidgets.utils.timeseries import get_timeseries_maxt, get_timeseries_mint
from .controllers import StartAndDurationController,  GroupAndSortController
from .ophys import RoiResponseSeriesWidget, TwoPhotonSeriesWidget
from .ecephys import ElectricalSeriesWidget


class AllenDashboard(widgets.VBox):
    def __init__(self, nwb):
        super().__init__()

        # self.tmin = get_timeseries_mint(time_series)
        # self.tmax = get_timeseries_maxt(time_series)
        self.lines_select = False

        self.btn_lines = widgets.Button(description='Enable spike times', button_style='')
        self.btn_lines.on_click(self.btn_lines_dealer)

        self.time_window_controller = StartAndDurationController(
            tmin=0,
            tmax=120,
            start=0,
            duration=5
        )

        self.electrical = ElectricalSeriesWidget(
            electrical_series=nwb.processing['ecephys'].data_interfaces['filtered_membrane_voltage'],
            foreign_time_window_controller=self.time_window_controller,
            dynamic_table_region_name=None,
            allen_dashboard=True
        )

        self.fluorescence = RoiResponseSeriesWidget(
            roi_response_series=nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'],
            foreign_time_window_controller=self.time_window_controller,
            foreign_group_and_sort_controller=None,
            dynamic_table_region_name=None,
            allen_dashboard=True
        )

        self.photon_series = TwoPhotonSeriesWidget(
            indexed_timeseries=nwb.acquisition['raw_ophys'],
            neurodata_vis_spec=None
        )

        header_box = widgets.VBox([self.time_window_controller, self.btn_lines])
        widgets_box = widgets.HBox([self.electrical, self.photon_series])

        self.output_box = widgets.VBox([header_box, widgets_box, self.fluorescence])

        self.children = [self.output_box]

    def btn_lines_dealer(self, b=0):
        self.lines_select = not self.lines_select
        if 'disable' in self.btn_lines.description.lower():
            self.btn_lines.description = 'Enable spike times'
        else:
            self.btn_lines.description = 'Disable spike times'