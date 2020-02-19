import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from nwbwidgets.ophys import show_grayscale_volume
from ndx_grayscalevolume import GrayscaleVolume
from nwbwidgets.view import default_neurodata_vis_spec

def test_show_grayscale_volume():
    vol = GrayscaleVolume(name='vol',data=np.random.rand(2700).reshape((30,30,3)))
    assert isinstance(show_grayscale_volume(vol, default_neurodata_vis_spec),widgets.Widget)
    
