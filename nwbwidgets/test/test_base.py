import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile
from ipywidgets import widgets
from nwbwidgets.view import default_neurodata_vis_spec
from nwbwidgets.base import df2grid_sps, df2grid_plot,show_neurodata_base

def test_df2grid_sps():
    df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                     columns=['a', 'b', 'c'])
    fig, big_ax, gs = df2grid_sps(df,'a','b')
    assert isinstance(fig,plt.Figure)
    assert isinstance(big_ax,plt.Subplot)
    assert isinstance(gs,plt.GridSpec)

  
def test_df2grid_plot():
    df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                     columns=['a', 'b', 'c'])
    def func(df,ax):
      return 1
    
    fig = df2grid_plot(df,'a','b',func)
    assert isinstance(fig,plt.Figure)
    
    
def test_show_neurodata_base():
    start_time = datetime(2017, 4, 3, 11, tzinfo=tzlocal())
    create_date = datetime(2017, 4, 15, 12, tzinfo=tzlocal())
    
    nwbfile = NWBFile(session_description='demonstrate NWBFile basics',  
                      identifier='NWB123',  
                      session_start_time=start_time,  
                      file_create_date=create_date,
                      related_publications='https://doi.org/10.1088/1741-2552/aaa904',
                      experimenter='Dr. Pack')
    
    assert isinstance(show_neurodata_base(nwbfile,default_neurodata_vis_spec), widgets.Widget)
