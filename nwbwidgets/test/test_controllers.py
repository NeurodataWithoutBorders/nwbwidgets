import ipywidgets as widgets
from nwbwidgets.controllers import float_range_controller,move_slider_down,move_slider_up,move_int_slider_down,move_int_slider_up,move_range_slider_down,move_range_slider_up
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
        assert(self.slider.value == 4)
    def test_move_slider_down_smaller(self):
        move_slider_down(self.slider,6)
        assert(self.slider.value == 2)
        
    def test_move_slider_up_bigger(self):
        move_slider_up(self.slider,2)
        assert(self.slider.value == 9)
    def test_move_slider_up_smaller(self):
        move_slider_up(self.slider,6)
        assert(self.slider.value == 4)
        
    def test_move_int_slider_down(self):
        move_int_slider_down(self.slider)
        assert(self.slider.value == 6)
        
    def test_move_int_slider_up(self):
        move_int_slider_up(self.slider)
        assert(self.slider.value == 8)


class RangeSliderTestCase(unittest.TestCase):

    def setUp(self):
        self.slider = widgets.IntRangeSlider(
        value=[5, 7],
        min=0,
        max=10,
        step=1,
        description='Test:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
        
    def test_move_range_slider_down_bigger(self):
        self.slider.value = (4, 6)
        move_range_slider_down(self.slider)
        assert(self.slider.value == (2, 4))
    def test_move_range_slider_down_smaller(self):
        self.slider.value = (2, 6)
        move_range_slider_down(self.slider)
        assert(self.slider.value == (0, 4))
    
    def test_move_range_slider_up_smaller(self):
        self.slider.value = (5, 7)
        move_range_slider_up(self.slider)
        assert(self.slider.value == (7, 9))
    def test_move_range_slider_up_bigger(self):
        self.slider.value = (5, 8)
        move_range_slider_up(self.slider)
        assert(self.slider.value == (7, 10))
