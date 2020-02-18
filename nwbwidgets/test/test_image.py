import numpy as np
import matplotlib.pyplot as plt
from pynwb.image import RGBImage,GrayscaleImage
from nwbwidgets.image import show_rbg_image,show_grayscale_image

def test_show_rbg_image():
    
    data = np.random.rand(2700).reshape((30,30,3))
    rgb_image = RGBImage(name='test_image',data=data)
    
    assert isinstance(show_rbg_image(rgb_image),plt.Figure)



def test_show_grayscale_image():
    
    data = np.random.rand(900).reshape((30,30))
    grayscale_image = GrayscaleImage(name='test_image',data=data)
    
    assert isinstance(show_grayscale_image(grayscale_image),plt.Figure)
