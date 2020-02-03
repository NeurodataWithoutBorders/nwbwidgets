import numpy as np

from pynwb import TimeSeries
from nwbwidgets.base import traces_widget


def test_timeseries_widget():
    ts = TimeSeries(name='name', description='no description',
                    data=np.array([[1., 2., 3., 4.],
                                   [11., 12., 13., 14.]]),
                    rate=100.)

    traces_widget(ts)



import numpy as np
import matplotlib.pyplot as plt
from nwbwidgets.base import fig2widget


def test_fig2widget():
    
    data = np.random.rand(160,3)
    
    fig=plt.figure(figsize=(10, 5))
    plt.plot(data)

    fig2widget(fig)




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
import matplotlib.pyplot as plt
from ipywidgets import widgets
from nwbwidgets.base import vis2widget


def test_vis2widget():
    
    #data = np.random.rand(160,3)
    
    #fig=plt.figure(figsize=(10, 5))
    #plt.plot(data)
    
    wg = widgets.IntSlider(
    value=7,
    min=0,
    max=10,
    step=1,
    description='Test:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d')

    #vis2widget(fig)
    vis2widget(wg)
