
import numpy as np
import pytest
import matplotlib.pyplot as plt
from nwbwidgets.base import fig2widget, vis2widget, show_subject, show_dynamic_table, dict2accordion
from ipywidgets import widgets
import pandas as pd
from hdmf.common import DynamicTable
from nwbwidgets.view import default_neurodata_vis_spec
from pynwb.misc import AnnotationSeries
from nwbwidgets.misc import show_annotations
from pynwb.file import Subject



def test_fig2widget():
    
    data = np.random.rand(160,3)
    
    fig=plt.figure(figsize=(10, 5))
    plt.plot(data)
    
    assert isinstance(fig2widget(fig), widgets.Widget)


class Test_vis2widget:
    
    def test_vis2widget_input_widget(self):

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

        assert isinstance(vis2widget(wg), widgets.Widget)

    def test_vis2widget_input_figure(self):

        data = np.random.rand(160,3)

        fig=plt.figure(figsize=(10, 5))
        plt.plot(data)

        assert isinstance(vis2widget(fig), widgets.Widget)
        
    def test_vis2widget_input_other(self):
        data = np.random.rand(160,3)
        with pytest.raises(ValueError, match="unsupported vis type"):
            vis2widget(data)


def test_show_subject():
    
    node = Subject(age='8', sex='m', species='macaque')
                   
    show_subject(node)

    
    
def test_show_dynamic_table():
    
    d = {'col1': [1, 2], 'col2': [3, 4]}
    DT = DynamicTable.from_dataframe(df=pd.DataFrame(data=d), 
                                     name='Test Dtable', 
                                     table_description='no description')
    
    show_dynamic_table(DT)



def test_dict2accordion():
    
    d = {'age': 8,'sex': 'm','species': 'macaque'}
    
    dict2accordion(d, default_neurodata_vis_spec)

    
    
def test_show_annotations():
    
    timestamps = np.array([0., 1., 2., 3., 4., 5., 6.])
    
    annotations = AnnotationSeries(name='test_annotations',timestamps=timestamps)

    show_annotations(annotations)

