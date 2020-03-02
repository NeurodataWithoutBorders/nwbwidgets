import numpy as np
from pynwb import TimeSeries
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile
from pynwb.epoch import TimeIntervals
from nwbwidgets.utils.timeseries import get_timeseries_tt,get_timeseries_maxt,get_timeseries_mint,get_timeseries_in_units, \
timeseries_time_to_ind,align_by_times,align_by_trials,align_by_trials,align_by_time_intervals
import unittest

  
def test_get_timeseries_tt_infstarting_time():
    
    data = list(range(100, 200, 10))
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=np.inf, rate=1.0)
    
    tt = get_timeseries_tt(ts)
    np.testing.assert_array_equal(tt,[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

    
def test_get_timeseries_tt_negativeistop():
    
    data = list(range(100, 200, 10))
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0., rate=1.0)
    
    tt = get_timeseries_tt(ts,istop=-1)
    np.testing.assert_array_equal(tt,[0., 1., 2., 3., 4., 5., 6., 7.])

    
def test_get_timeseries_in_units():
    data = list(range(100, 200, 10))
    timestamps = list(range(10))
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', timestamps=timestamps,conversion=np.inf)
    data,unit = get_timeseries_in_units(ts)
    assert(unit is None)
    assert(data==[100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
    
    
def test_align_by_trials():
    start_time = datetime(2017, 4, 3, 11, tzinfo=tzlocal())
    create_date = datetime(2017, 4, 15, 12, tzinfo=tzlocal())

    nwbfile = NWBFile(session_description='NWBFile for PSTH', 
                      identifier='NWB123',  
                      session_start_time=start_time,  
                      file_create_date=create_date)
    
    data = list(range(100, 200, 10))
    timestamps = list(range(10))
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', timestamps=timestamps)
    nwbfile.add_acquisition(ts)
    
    nwbfile.add_trial_column(name='stim', description='the visual stimuli during the trial')
    nwbfile.add_trial(start_time=0.0, stop_time=2.0, stim='person')
    nwbfile.add_trial(start_time=3.0, stop_time=5.0, stim='ocean')
    nwbfile.add_trial(start_time=6.0, stop_time=8.0, stim='desert')
    
    np.testing.assert_array_equal(align_by_trials(ts),np.array([[110],[140],[170]]))
    
    
    
     
class TimeSeriesTimeStampTestCase(unittest.TestCase):

    def setUp(self):
        data = list(range(100, 200, 10))
        timestamps = list(range(10))
        self.ts = TimeSeries(name='test_timeseries', data=data, unit='m', timestamps=timestamps)
    
    def test_get_timeseries_maxt(self):
        maxt = get_timeseries_maxt(self.ts)
        assert(maxt==9)
      
    def test_get_timeseries_mint(self):
        mint = get_timeseries_mint(self.ts)
        assert(mint==0)

    def test_timeseries_time_to_ind(self):
        assert(timeseries_time_to_ind(self.ts,3)==4)

    def test_align_by_times(self):
        assert(np.array_equal(align_by_times(self.ts,[0,1,2],[4,5,6]),np.array([[110, 120, 130, 140],[120, 130, 140, 150],[130, 140, 150, 160]])))
    
    def test_get_timeseries_tt_timestamp(self):
        tt = get_timeseries_tt(self.ts)
        np.testing.assert_array_equal(tt,[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        
    def test_align_by_time_intervals(self):
        intervals=TimeIntervals(name='Time Intervals')
        np.testing.assert_array_equal(align_by_time_intervals(timeseries=self.ts,intervals=intervals,stop_label=None),np.array([]))
        
    
