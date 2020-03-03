import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile
from pynwb.misc import Units, DecompositionSeries, AnnotationSeries
from ipywidgets import widgets
from nwbwidgets.misc import show_psth, psth_widget, show_decomposition_traces, show_decomposition_series, raster_widget, \
show_session_raster,show_annotations
import unittest


def test_show_psth():
    data = np.random.random([6, 50])
    colors = ['C{}'.format(i) for i in range(6)]
    fig, ax = plt.subplots()
    assert isinstance(show_psth(data=data, colors=colors, ax=ax, before=0, after=1),plt.Subplot)
    

def test_show_annotations():
    timestamps = np.array([0., 1., 2., 3., 4., 5., 6.])
    annotations = AnnotationSeries(name='test_annotations',timestamps=timestamps)
    show_annotations(annotations)
    
    
class ShowPSTHTestCase(unittest.TestCase):

    def setUp(self):
        """
        Trials must exist.
        """
        start_time = datetime(2017, 4, 3, 11, tzinfo=tzlocal())
        create_date = datetime(2017, 4, 15, 12, tzinfo=tzlocal())

        self.nwbfile = NWBFile(session_description='NWBFile for PSTH', 
                          identifier='NWB123',  
                          session_start_time=start_time,  
                          file_create_date=create_date)
        
        self.nwbfile.add_unit_column('location', 'the anatomical location of this unit')
        self.nwbfile.add_unit_column('quality', 'the quality for the inference of this unit')
        
        self.nwbfile.add_unit(id=1, spike_times=[2.2, 3.0, 4.5],
                         obs_intervals=[[1, 10]], location='CA1', quality=0.95)
        self.nwbfile.add_unit(id=2, spike_times=[2.2, 3.0, 25.0, 26.0],
                         obs_intervals=[[1, 10], [20, 30]], location='CA3', quality=0.85)
        self.nwbfile.add_unit(id=3, spike_times=[1.2, 2.3, 3.3, 4.5],
                         obs_intervals=[[1, 10], [20, 30]], location='CA1', quality=0.90)
        
        self.nwbfile.add_trial_column(name='stim', description='the visual stimuli during the trial')

        self.nwbfile.add_trial(start_time=0.0, stop_time=2.0, stim='person')
        self.nwbfile.add_trial(start_time=3.0, stop_time=5.0, stim='ocean')
        self.nwbfile.add_trial(start_time=6.0, stop_time=8.0, stim='desert')

    def test_psth_widget(self):
        assert isinstance(psth_widget(self.nwbfile.units),widgets.Widget)
        
    def test_raster_widget(self):
        assert isinstance(raster_widget(self.nwbfile.units),widgets.Widget)
        
    def test_show_session_raster(self):
        assert isinstance(show_session_raster(self.nwbfile.units),plt.Figure)

    
class ShowDecompositionTestCase(unittest.TestCase):

    def setUp(self):
    
        data = np.random.rand(160,2,3)

        self.ds = DecompositionSeries(name='Test Decomposition',data=data,
                                     metric='amplitude',rate=1.0)
        
    def test_show_decomposition_traces(self):

        assert isinstance(show_decomposition_traces(self.ds),widgets.Widget)
        
    def test_show_decomposition_series(self):
        
        assert isinstance(show_decomposition_series(self.ds),widgets.Widget)
