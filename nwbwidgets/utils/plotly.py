import plotly.graph_objects as go
import numpy as np
from ipywidgets import widgets


def multi_trace(x, y, color, label=None, fig=None):
    """ Create multiple traces that are associated with a single legend label

    Parameters
    ----------
    x: array-like
    y: array-like
    color: str
    label: str, optional
    fig: go.FigureWidget

    Returns
    -------

    """
    if fig is None:
        fig = go.FigureWidget()

    for i, yy in enumerate(y):
        if label is not None and i:
            showlegend = False
        else:
            showlegend = True

        fig.add_scatter(x=x, y=yy, legendgroup=label, name=label, showlegend=showlegend, line={'color': color})

    return fig


def event_group(times_list, offset=0, color='Black', label=None, fig=None, marker=None, line_width=None):
    """ Create an event raster that are all associated with a single legend label

    Parameters
    ----------
    times_list: list of array-like
    offset: float, optional
    label: str, optional
    fig: go.FigureWidget

    optional, passed to go.Scatter.marker:
    marker: str
    line_width: str
    color: str
        default: Black


    Returns
    -------

    """
    if fig is None:
        fig = go.FigureWidget()

    if label is not None:
        showlegend = True
    else:
        showlegend = False

    for i, times in enumerate(times_list):
        if len(times):
            fig.add_scatter(x=times, y=np.ones_like(times) * (i + offset),
                            marker=dict(color=color, line_width=line_width, symbol=marker, line_color=color),
                            legendgroup=str(label),
                            name=label,
                            showlegend=showlegend,
                            mode='markers'
                            )
            showlegend = False

    return fig


class Peakaboo:
    """Make a plotly figure disappear as it is being updated"""

    def __init__(self, container, index, placeholder=widgets.HTML('Rendering...')):
        self.container = container
        self.index = index
        self.placeholder = placeholder

    def __enter__(self):

        self.children = list(self.container.children)
        self.fig = self.children[self.index]
        self.children[self.index] = self.placeholder
        self.container.children = self.children

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.children[self.index] = self.fig
        self.container.children = self.children
