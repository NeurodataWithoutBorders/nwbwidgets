
import numpy as np
import matplotlib.pyplot as plt
from nwbwidgets.base import fig2widget


def test_fig2widget():
    
    data = np.random.rand(160,3)
    
    fig=plt.figure(figsize=(10, 5))
    plt.plot(data)

    fig2widget(fig)
    
    isinstance(fig2widget(fig), widgets.Widget)



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