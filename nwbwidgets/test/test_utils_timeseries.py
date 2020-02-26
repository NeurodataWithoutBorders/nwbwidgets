import numpy as np
from pynwb import TimeSeries
from nwbwidgets.utils.timeseries import get_timeseries_tt,get_timeseries_maxt,get_timeseries_mint
import unittest



def test_get_timeseries_tt_timestamp():
    
    data = list(range(100, 200, 10))
    timestamps = list(range(10))
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', timestamps=timestamps)
    
    tt = get_timeseries_tt(ts)
    np.testing.assert_array_equal(tt,[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

    
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
    
    
def test_get_timeseries_maxt():
    data = list(range(100, 200, 10))
    timestamps = list(range(10))
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', timestamps=timestamps)
    
    maxt = get_timeseries_maxt(ts)
    assert(maxt==9)
    
    
def test_get_timeseries_mint():
    data = list(range(100, 200, 10))
    timestamps = list(range(10))
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', timestamps=timestamps)
    
    mint = get_timeseries_mint(ts)
    assert(mint==0)
