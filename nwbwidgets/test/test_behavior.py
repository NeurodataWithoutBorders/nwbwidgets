import numpy as np
from pynwb import TimeSeries
from pynwb.behavior import Position, SpatialSeries, BehavioralEvents
from nwbwidgets.view import default_neurodata_vis_spec
from nwbwidgets.behavior import show_position, show_behavioral_events, show_spatial_series_over_time, show_spatial_series
import unittest

class ShowSpatialSeriesTestCase(unittest.TestCase):

    def setUp(self):
        self.spatial_series = SpatialSeries(name = 'position',
                                   data = np.linspace(0, 1, 20),
                                   rate = 50.,
                                   reference_frame = 'starting gate')
    def test_show_position(self):
 
        position = Position(spatial_series = self.spatial_series)

        show_position(position, default_neurodata_vis_spec)

    def test_show_spatial_series_over_time(self):

        show_spatial_series_over_time(self.spatial_series)

    def test_show_spatial_series(self):

        show_spatial_series(self.spatial_series)

        
class ShowSpatialSeriesTwoDTestCase(unittest.TestCase):

    def setUp(self):
        self.spatial_series = SpatialSeries(name = 'position',
                                       data = np.array([np.linspace(0, 1, 20), np.linspace(0, 1, 20)]).T,
                                       rate = 50.,
                                       reference_frame = 'starting gate')

    def test_show_spatial_series_over_time_twoD(self):
        
        show_spatial_series_over_time(self.spatial_series)

    def test_show_spatial_series_twoD(self):

        show_spatial_series(self.spatial_series)
    

class ShowSpatialSeriesThreeDTestCase(unittest.TestCase):

    def setUp(self):
        self.spatial_series = SpatialSeries(name = 'position',
                                       data = np.array([np.linspace(0, 1, 20), np.linspace(0, 1, 20),np.linspace(0, 1, 20)]).T,
                                       rate = 50.,
                                       reference_frame = 'starting gate')

    def test_show_spatial_series_over_time_threeD(self):
        
        show_spatial_series_over_time(self.spatial_series)

    def test_show_spatial_series_threeD(self):

        show_spatial_series(self.spatial_series)

        
def test_show_behavioral_events():
    
    data = list(range(100, 200, 10))
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)
    
    beh_events = BehavioralEvents(time_series=ts)
    
    show_behavioral_events(beh_events, default_neurodata_vis_spec)
