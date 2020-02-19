import ipywidgets as widgets
from nwbwidgets.controllers import float_range_controller,move_slider_down,move_slider_up,move_int_slider_down,move_int_slider_up
import unittest

def test_float_range_controller():

    assert isinstance(float_range_controller(tmin=1,tmax=26),widgets.Widget)

    
class MoveSliderTestCase(unittest.TestCase):

    def setUp(self):
        self.slider = widgets.IntSlider(
        value=7,
        min=2,
        max=10,
        step=1,
        description='Test:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
    
    def test_move_slider_down_bigger(self):
        move_slider_down(self.slider,3)
    def test_move_slider_down_smaller(self):
        move_slider_down(self.slider,6)
        
    def test_move_slider_up_bigger(self):
        move_slider_up(self.slider,2)
    def test_move_slider_up_smaller(self):
        move_slider_up(self.slider,6)
        
    def test_move_int_slider_down(self):
        move_int_slider_down(self.slider)
        
    def test_move_int_slider_up(self):
        move_int_slider_up(self.slider)
