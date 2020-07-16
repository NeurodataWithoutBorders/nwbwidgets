from ipywidgets import widgets
from pynwb import TimeSeries
#from nwbwidgets.utils.timeseries import get_timeseries_maxt, get_timeseries_mint
from .controllers import StartAndDurationController,  GroupAndSortController
from .ophys import RoiResponseSeriesWidget
from .ecephys import ElectricalSeriesWidget


class AllenDashboard(widgets.VBox):
    def __init__(self, nwb):
        super().__init__()

        # self.tmin = get_timeseries_mint(time_series)
        # self.tmax = get_timeseries_maxt(time_series)
        self.time_window_controller = StartAndDurationController(
            tmin=0,
            tmax=120,
            start=0,
            duration=5
        )

        self.electrical = ElectricalSeriesWidget(
            electrical_series=nwb.processing['ecephys'].data_interfaces['filtered_membrane_voltage'],
            foreign_time_window_controller=self.time_window_controller,
            dynamic_table_region_name=None
        )

        self.fluorescence = RoiResponseSeriesWidget(
            roi_response_series=nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'],
            foreign_time_window_controller=self.time_window_controller,
            foreign_group_and_sort_controller=None,
            dynamic_table_region_name=None
        )

        self.output_box = widgets.VBox([self.time_window_controller, self.electrical, self.fluorescence])

        self.children = [self.output_box]
