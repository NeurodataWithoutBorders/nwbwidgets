import matplotlib.pyplot as plt
import numpy as np
from nwbwidgets.utils.mpl import create_big_ax,grid_sps



def test_create_big_ax():
    fig = plt.figure(tight_layout=True)
    plt.plot(np.arange(0, 1e6, 1000))
    assert isinstance(create_big_ax(fig),plt.Subplot)

def test_grid_sps():
    shape = (5,6)
    fig, big_ax, gs = grid_sps(shape)
    assert isinstance(fig,plt.Figure)
    assert isinstance(big_ax,plt.Subplot)
    assert isinstance(gs,plt.GridSpec)
    
