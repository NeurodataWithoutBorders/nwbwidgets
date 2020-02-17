import numpy as np
from pynwb import TimeSeries
from pynwb.behavior import Position, SpatialSeries, BehavioralEvents
from nwbwidgets.view import default_neurodata_vis_spec
from nwbwidgets.behavior import show_position, show_behavioral_events, show_spatial_series_over_time, show_spatial_series

def test_show_position():
    
    spatial_series = SpatialSeries(name = 'position',
                               data = np.linspace(0, 1, 20),
                               rate = 50.,
                               reference_frame = 'starting gate')
    position = Position(spatial_series = spatial_series)
    

    show_position(position, default_neurodata_vis_spec)



def test_show_behavioral_events():
    
    data = list(range(100, 200, 10))
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)
    
    beh_events = BehavioralEvents(time_series=ts)
    
    show_behavioral_events(beh_events, default_neurodata_vis_spec)

    
    
def test_show_spatial_series_over_time():
    
    spatial_series = SpatialSeries(name = 'position',
                               data = np.linspace(0, 1, 20),
                               rate = 50.,
                               reference_frame = 'starting gate')
    

    show_spatial_series_over_time(spatial_series)

    
    
def test_show_spatial_series():
    
    spatial_series = SpatialSeries(name = 'position',
                               data = np.linspace(0, 1, 20),
                               rate = 50.,
                               reference_frame = 'starting gate')
    

    show_spatial_series(spatial_series)
    
    
