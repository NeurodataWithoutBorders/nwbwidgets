import ipywidgets as widgets
from nwbwidgets.controllers import float_range_controller

def test_float_range_controller():

    assert isinstance(float_range_controller(tmin=1,tmax=26),widgets.Widget)
