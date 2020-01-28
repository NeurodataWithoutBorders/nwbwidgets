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
    
    data = np.array([[1., 2., 3., 4., 1., 2., 3., 4.],
                               [11., 12., 13., 14., 11., 12., 13., 14.]])
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)

    
    show_text_fields(ts)
