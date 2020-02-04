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
    
    tt = get_timeseries_tt(ts)
    np.testing.assert_array_equal(tt,[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

    
    

import numpy as np
from pynwb import TimeSeries
from nwbwidgets.base import show_text_fields


def test_show_text_fields():
    
    data = np.random.rand(160,3)
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)

    
    show_text_fields(ts)



import numpy as np
from pynwb import TimeSeries
from nwbwidgets.base import show_ts_fields


def test_show_ts_fields():
    
    data = np.random.rand(160,3)
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)

    
    show_ts_fields(ts)



import numpy as np
from pynwb import TimeSeries
from nwbwidgets.base import show_timeseries


def test_show_timeseries():
    
    data = np.random.rand(160,3)
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)

    
    show_timeseries(ts, istart=5, istop=56)
