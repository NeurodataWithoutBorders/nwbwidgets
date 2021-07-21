import plotly.graph_objects as go
import numpy as np


def multi_trace(x, y, color, label=None, fig=None, insert_nans=False):
    """Create multiple traces that are associated with a single legend label

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

    if insert_nans:
        y_nans = []
        x_nans = []
        for xx,yy in zip(x,y):
            y_nans.append(np.append(yy,np.nan))
            x_nans.append(np.append(xx, np.nan))
        y_plot = np.concatenate(y_nans,axis=0)
        x_plot = np.concatenate(x_nans, axis=0)
        fig.add_scattergl(
            x=x_plot,
            y=y_plot,
            name=label,
            line={"color": color},
        )
        return fig
    else:
        for i, yy in enumerate(y):
            if label is not None and i:
                showlegend = False
            else:
                showlegend = True

            fig.add_scattergl(
                x=x,
                y=yy,
                legendgroup=label,
                name=label,
                showlegend=showlegend,
                line={"color": color},
            )

        return fig


def event_group(
    times_list,
    offset=0,
    color="Black",
    label=None,
    fig=None,
    marker=None,
    line_width=None,
):
    """Create an event raster that are all associated with a single legend label

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
            fig.add_scattergl(
                x=times,
                y=np.ones_like(times) * (i + offset),
                marker=dict(
                    color=color, line_width=line_width, symbol=marker, line_color=color
                ),
                legendgroup=str(label),
                name=label,
                showlegend=showlegend,
                mode="markers",
            )
            showlegend = False

    return fig
