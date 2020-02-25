import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nwbwidgets.base import df2grid_sps

def test_df2grid_sps():
  df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                   columns=['a', 'b', 'c'])
  fig, big_ax, gs = df2grid_sps(df,'a','b')
  assert isinstance(fig,plt.Figure)
  assert isinstance(big_ax,plt.Subplot)
  assert isinstance(gs,plt.GridSpec)
