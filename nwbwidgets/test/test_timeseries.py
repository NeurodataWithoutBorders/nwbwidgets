import numpy as np
from pynwb import TimeSeries
from ipywidgets import widgets
from nwbwidgets.base import traces_widget, show_text_fields
from nwbwidgets.timeseries import traces_widget,show_ts_fields, show_timeseries
from nwbwidgets.utils.timeseries import get_timeseries_tt



def test_get_timeseries_tt():
    
    data = list(range(100, 200, 10))
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)
    
    tt = get_timeseries_tt(ts)
    np.testing.assert_array_equal(tt,[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])


def test_timeseries_widget():
    ts = TimeSeries(name='name', description='no description',
                    data=np.array([[1., 2., 3., 4.],
                                   [11., 12., 13., 14.]]),
                    rate=100.)

    traces_widget(ts)
    
class ShowTimeSeriesTestCase(unittest.TestCase):
    
    def setUp(self):
        data = np.random.rand(160,3)
        self.ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)
        
    def test_show_text_fields():
        assert isinstance(show_text_fields(self.ts), widgets.Widget)
        

    def test_show_ts_fields():
        assert isinstance(show_ts_fields(self.ts), widgets.Widget)
        
    def test_show_timeseries():
        assert isinstance(show_timeseries(self.ts, istart=5, istop=56), widgets.Widget)
        
