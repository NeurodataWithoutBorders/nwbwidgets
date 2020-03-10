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


def grid_sps(shape, subplot_spec=None, fig=None):
    """
    Create subplot_spec from pandas.DataFrame

    Parameters
    ----------
    shape: tuple
    subplot_spec: GridSpec, optional
    fig: matplotlib.pyplot.Figure, optional

    Returns
    -------

    """

    if fig is None:
        fig = plt.gcf()

    if subplot_spec is not None:
        gs = gridspec.GridSpecFromSubplotSpec(shape[0], shape[1], subplot_spec=subplot_spec)
        big_ax = plt.Subplot(fig, subplot_spec)
    else:
        gs = gridspec.GridSpec(shape[0], shape[1], figure=fig)
        big_ax = plt.Subplot(fig, gridspec.GridSpec(1, 1)[0])
    fig.add_subplot(big_ax)

    [sp.set_visible(False) for sp in big_ax.spines.values()]
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.patch.set_facecolor('none')

    return fig, big_ax, gs

