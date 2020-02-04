
import numpy as np
import matplotlib.pyplot as plt
from nwbwidgets.base import fig2widget


def test_fig2widget():
    
    data = np.random.rand(160,3)
    
    fig=plt.figure(figsize=(10, 5))
    plt.plot(data)
    
    isinstance(fig2widget(fig), widgets.Widget)




import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from nwbwidgets.base import vis2widget


def test_vis2widget_input_widget():
        
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

    vis2widget(wg)


def test_vis2widget_input_figure():
    
    data = np.random.rand(160,3)
    
    fig=plt.figure(figsize=(10, 5))
    plt.plot(data)
    
    vis2widget(fig)



import numpy as np
from pynwb import TimeSeries
from nwbwidgets.base import show_subject


def test_show_subject():
    
    data = np.random.rand(160,3)
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)

    
    show_subject(ts)
