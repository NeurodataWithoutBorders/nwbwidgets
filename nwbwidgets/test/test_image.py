import numpy as np
import matplotlib.pyplot as plt
from pynwb.image import RGBImage
from nwbwidgets.image import show_rbg_image

def test_show_rbg_image():
    
    data = np.random.rand(2700).reshape((30,30,3))
    rgb_image = RGBImage(name='test_image',data=data)
    
    assert isinstance(show_rbg_image(rgb_image),plt.Figure)
