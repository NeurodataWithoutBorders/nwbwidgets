
import numpy as np
from pynwb import NWBFile
from nwbwidgets.base import processing_module, nwb2widget, show_neurodata_base
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb.behavior import Position, SpatialSeries
from pynwb import ProcessingModule
from nwbwidgets.view import default_neurodata_vis_spec
import unittest


class ProcessingModuleTestCase(unittest.TestCase):
    def setUp(self):
        spatial_series = SpatialSeries(name='position',
                                   data=np.linspace(0, 1, 20),
                                   rate=50.,
                                   reference_frame='starting gate')
        self.position = Position(spatial_series=spatial_series)
    
    def test_processing_module(self):

        start_time = datetime(2020, 1, 29, 11, tzinfo=tzlocal())
        nwbfile = NWBFile(session_description='Test Session',  
                      identifier='NWBPM',  
                      session_start_time=start_time)

        behavior_module = ProcessingModule(name='behavior',
                                                       description='preprocessed behavioral data')
        nwbfile.add_processing_module(behavior_module)

        nwbfile.processing['behavior'].add(self.position)

        self.processing_module(nwbfile.processing['behavior'], default_neurodata_vis_spec)

    def test_nwb2widget(self):

        self.nwb2widget(self.position, default_neurodata_vis_spec)


    def test_show_neurodata_base(self):

        self.show_neurodata_base(self.position, default_neurodata_vis_spec)


