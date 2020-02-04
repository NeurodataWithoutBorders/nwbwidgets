
import numpy as np
from pynwb import NWBFile
from nwbwidgets.base import processing_module, nwb2widget, show_neurodata_base
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb.behavior import Position, SpatialSeries
from pynwb import ProcessingModule
from nwbwidgets.view import default_neurodata_vis_spec



def test_processing_module():
    
    # Create a NWB file
    start_time = datetime(2020, 1, 29, 11, tzinfo=tzlocal())
    nwbfile = NWBFile(session_description='Test Session',  
                  identifier='NWBPM',  
                  session_start_time=start_time)
    
    # Create a data interface object (position)
    spatial_series = SpatialSeries(name='position',
                               data=np.linspace(0, 1, 20),
                               rate=50.,
                               reference_frame='starting gate')
    position = Position(spatial_series=spatial_series)
    
    # Create a processing module
    behavior_module = ProcessingModule(name='behavior',
                                                   description='preprocessed behavioral data')
    nwbfile.add_processing_module(behavior_module)
    
    # Add data interface to the processing module
    nwbfile.processing['behavior'].add(position)
    
    # Test processing_module function in base.py
    processing_module(nwbfile.processing['behavior'], default_neurodata_vis_spec)


def test_nwb2widget():
    
    # Create a NWB file
    #start_time = datetime(2020, 1, 29, 11, tzinfo=tzlocal())
    #nwbfile = NWBFile(session_description = 'Test Session',  
    #              identifier = 'NWBPM',  
    #              session_start_time = start_time)
    
    # Create a data interface object (position)
    spatial_series = SpatialSeries(name='position',
                               data=np.linspace(0, 1, 20),
                               rate=50.,
                               reference_frame='starting gate')
    position = Position(spatial_series=spatial_series)
    
    # Create a processing module
    #behavior_module = ProcessingModule(name='behavior',
    #                                               description='preprocessed behavioral data')
    #nwbfile.add_processing_module(behavior_module)
    
    # Add data interface to the processing module
    #nwbfile.processing['behavior'].add(position)

    # Test processing_module function in base.py
    #nwb2widget(nwbfile.processing['behavior'].data_interfaces, neurodata_vis_spec)
    nwb2widget(position, default_neurodata_vis_spec)
    
    
def test_show_neurodata_base():
    
    
    # Create a data interface object (position)
    spatial_series = SpatialSeries(name = 'position',
                               data = np.linspace(0, 1, 20),
                               rate = 50.,
                               reference_frame = 'starting gate')
    position = Position(spatial_series = spatial_series)
    
    
    # Test processing_module function in base.py
    show_neurodata_base(position, default_neurodata_vis_spec)
