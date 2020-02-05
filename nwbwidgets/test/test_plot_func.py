
import numpy as np
import matplotlib.pyplot as plt
from nwbwidgets.base import fig2widget, vis2widget, show_subject, import show_dynamic_table
from ipywidgets import widgets
from pynwb import TimeSeries
import pandas as pd
from hdmf.common import DynamicTable
from pynwb.behavior import Position, SpatialSeries
from nwbwidgets.view import default_neurodata_vis_spec
from nwbwidgets.behavior import show_position




def test_fig2widget():
    
    data = np.random.rand(160,3)
    
    fig=plt.figure(figsize=(10, 5))
    plt.plot(data)
    
    isinstance(fig2widget(fig), widgets.Widget)


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


def test_show_subject():
    
    data = np.random.rand(160,3)
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)

    
    show_subject(ts)

    
    
def test_show_dynamic_table():
    
    d = {'col1': [1, 2], 'col2': [3, 4]}
    DT = DynamicTable.from_dataframe(df=pd.DataFrame(data=d), 
                                     name='Test Dtable', 
                                     table_description='no description')
    
    show_dynamic_table(DT)



def test_show_position():
    
    
    # Create a data interface object (position)
    spatial_series = SpatialSeries(name = 'position',
                               data = np.linspace(0, 1, 20),
                               rate = 50.,
                               reference_frame = 'starting gate')
    position = Position(spatial_series = spatial_series)
    
    
    # Test show position function in behavior.py
    show_position(position, default_neurodata_vis_spec)
