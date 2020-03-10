import matplotlib.pyplot as plt
from matplotlib import gridspec


def create_big_ax(fig):
    big_ax = plt.Subplot(fig, gridspec.GridSpec(1, 1)[0])
    fig.add_subplot(big_ax)
    [sp.set_visible(False) for sp in big_ax.spines.values()]
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.patch.set_facecolor('none')

    return big_ax

