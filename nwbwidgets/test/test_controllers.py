from nwbwidgets.controllers import RangeController
import unittest


class FloatRangeControllerTestCase(unittest.TestCase):

    def setUp(self):
        self.range_controller = RangeController(vmin=0, vmax=10, start_value=(5, 7))
        
    def test_move_range_slider_down_bigger(self):
        self.range_controller.value = (4, 6)
        self.range_controller.move_down('filler')
        assert(self.range_controller.value == (2, 4))

    def test_move_range_slider_down_smaller(self):
        self.range_controller.value = (2, 6)
        self.range_controller.move_down('filler')
        assert(self.range_controller.value == (0, 4))
    
    def test_move_range_slider_up_smaller(self):
        self.range_controller.value = (5, 7)
        self.range_controller.move_up('filler')
        assert(self.range_controller.value == (7, 9))

    def test_move_range_slider_up_bigger(self):
        self.range_controller.value = (5, 8)
        self.range_controller.move_up('filler')
        assert(self.range_controller.value == (7, 10))
