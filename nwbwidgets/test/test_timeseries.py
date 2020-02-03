import numpy as np

from pynwb import TimeSeries
from nwbwidgets.base import traces_widget


def test_timeseries_widget():
    ts = TimeSeries(name='name', description='no description',
                    data=np.array([[1., 2., 3., 4.],
                                   [11., 12., 13., 14.]]),
                    rate=100.)

    traces_widget(ts)



from pynwb import TimeSeries
from nwbwidgets.utils.timeseries import get_timeseries_tt


def test_get_timeseries_tt():
    
    data = list(range(100, 200, 10))
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)
    
    get_timeseries_tt(ts)

    
    
import numpy as np
from pynwb import TimeSeries
from nwbwidgets.base import show_text_fields


def test_show_text_fields():
    
    data = np.random.rand(160,3)
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)

    
    show_text_fields(ts)

    

import numpy as np
from pynwb import NWBFile
from nwbwidgets.base import processing_module
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb.behavior import Position
from pynwb.behavior import SpatialSeries
from pynwb import ProcessingModule



def test_processing_module():
    
    # Create a NWB file
    start_time = datetime(2020, 1, 29, 11, tzinfo=tzlocal())
    nwbfile = NWBFile(session_description = 'Test Session',  
                  identifier = 'NWBPM',  
                  session_start_time = start_time)
    
    # Create a data interface object (position)
    spatial_series = SpatialSeries(name = 'position',
                               data = np.linspace(0, 1, 20),
                               rate = 50.,
                               reference_frame = 'starting gate')
    position = Position(spatial_series = spatial_series)
    
    # Create a processing module
    behavior_module = ProcessingModule(name='behavior',
                                                   description='preprocessed behavioral data')
    nwbfile.add_processing_module(behavior_module)
    
    # Add data interface to the processing module
    nwbfile.processing['behavior'].add(position)
    
    # Define neurodata_vis_spec
    neurodata_vis_spec =	{'ndtype': 'str'}
    
    # Test processing_module function in base.py
    processing_module(nwbfile.processing['behavior'], neurodata_vis_spec)
