import numpy as np
from pynwb import TimeSeries
from nwbwidgets.utils.timeseries import get_timeseries_tt,get_timeseries_maxt,get_timeseries_mint,get_timeseries_in_units,timeseries_time_to_ind,align_by_times,align_by_trials
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
    
