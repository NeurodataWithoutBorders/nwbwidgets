import numpy as np
from pynwb import TimeSeries
from nwbwidgets.utils.timeseries import get_timeseries_tt
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
