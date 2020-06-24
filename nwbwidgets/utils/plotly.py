import plotly.graph_objects as go
import numpy as np
from ipywidgets import widgets


def multi_trace(x, y, color, label=None, fig=None):
    if fig is None:
        fig = go.FigureWidget()

    for i, yy in enumerate(y):
        if label is not None and i:
            showlegend = False
        else:
            showlegend = True

        fig.add_scatter(x=x, y=yy, legendgroup=label, name=label, showlegend=showlegend, line={'color': color})


def event_group(times_list, offset=0, color='Black', label=None, fig=None):
    if fig is None:
        fig = go.FigureWidget()

    if label is not None:
        showlegend = True
    else:
        showlegend = False

    for i, times in enumerate(times_list):
        if len(times):
            fig.add_scatter(x=times, y=np.ones_like(times) * (i + offset),
                            marker=dict(color=color, line_width=2, symbol='line-ns', line_color=color),
                            legendgroup=str(label),
                            name=label,
                            showlegend=showlegend,
                            mode='markers'
                            )
            showlegend = False
    return fig


class Peakaboo():

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
