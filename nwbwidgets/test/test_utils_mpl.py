import matplotlib.pyplot as plt
import numpy as np
from nwbwidgets.utils.mpl import create_big_ax,grid_sps


def test_create_big_ax():
    fig = plt.figure(tight_layout=True)
    plt.plot(np.arange(0, 1e6, 1000))
    assert isinstance(create_big_ax(fig),plt.Subplot)

def test_grid_sps():
    shape = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    grid_sps(shape)
