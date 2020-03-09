from nwbwidgets.utils.cmaps import linear_transfer_function,matplotlib_transfer_function
from ipyvolume import TransferFunction

def test_linear_transfer_function():
    assert isinstance(linear_transfer_function('blue',reverse_opacity=True),TransferFunction)
    
def test_matplotlib_transfer_function():
    assert isinstance(matplotlib_transfer_function('PuBuGn_r'),TransferFunction)
