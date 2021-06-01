import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import widgets
from nwbwidgets import base
from plotly import graph_objects as go
from pynwb.behavior import SpatialSeries, BehavioralEvents

from .utils.timeseries import get_timeseries_tt, get_timeseries_in_units

from .timeseries import (
    AlignMultiTraceTimeSeriesByTrialsConstant,
    AlignMultiTraceTimeSeriesByTrialsVariable,
)


def show_behavioral_events(beh_events: BehavioralEvents, neurodata_vis_spec: dict):
    return base.dict2accordion(
        beh_events.time_series, neurodata_vis_spec, ls="", marker="|"
    )


def show_spatial_series_over_time(node: SpatialSeries, **kwargs):
    text_widget = base.show_text_fields(
        node, exclude=("timestamps_unit", "comments", "data", "timestamps", "interval")
    )

    data, unit = get_timeseries_in_units(node)

    if len(data.shape) == 1:
        ndims = 1
    else:
        ndims = data.shape[1]

    tt = get_timeseries_tt(node)

    if ndims == 1:
        fig, ax = plt.subplots()
        ax.plot(tt, data, **kwargs)
        ax.set_xlabel("t (sec)")
        if unit:
            ax.set_ylabel("x ({})".format(unit))
        else:
            ax.set_ylabel("x")

    else:
        fig, axs = plt.subplots(ndims, 1, sharex=True)

        for i, (ax, dim_label) in enumerate(zip(axs, ("x", "y", "z"))):
            ax.plot(tt, data[:, i], **kwargs)
            if unit:
                ax.set_ylabel(dim_label + " ({})".format(unit))
            else:
                ax.set_ylabel(dim_label)
        ax.set_xlabel("t (sec)")

    return widgets.HBox([text_widget, base.fig2widget(fig)])


def show_spatial_series(node: SpatialSeries, **kwargs):
    data, unit = get_timeseries_in_units(node)
    tt = get_timeseries_tt(node)

    if len(data.shape) == 1:
        fig, ax = plt.subplots()
        ax.plot(tt, data, **kwargs)
        ax.set_xlabel("t (sec)")
        if unit:
            ax.set_xlabel("x ({})".format(unit))
        else:
            ax.set_xlabel("x")
        ax.set_ylabel("x")

    elif data.shape[1] == 2:
        fig, ax = plt.subplots()
        ax.plot(data[:, 0], data[:, 1], **kwargs)
        if unit:
            ax.set_xlabel("x ({})".format(unit))
            ax.set_ylabel("y ({})".format(unit))
        else:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        ax.axis("equal")

    elif data.shape[1] == 3:
        import ipyvolume.pylab as p3

        fig = p3.figure()
        p3.scatter(data[:, 0], data[:, 1], data[:, 2], **kwargs)
        p3.xlim(np.min(data[:, 0]), np.max(data[:, 0]))
        p3.ylim(np.min(data[:, 1]), np.max(data[:, 1]))
        p3.zlim(np.min(data[:, 2]), np.max(data[:, 2]))

    else:
        raise NotImplementedError

    return fig


def plotly_show_spatial_trace(node):
    data, unit = get_timeseries_in_units(node)
    tt = get_timeseries_tt(node)

    fig = go.FigureWidget()

    if len(data.shape) == 1:
        fig.add_trace(go.Scatter(x=tt, y=data))
        fig.update_xaxes(title_text="time (s)")
        if unit:
            fig.update_yaxes(title_text="x ({})".format(unit))
        else:
            fig.update_yaxes(title_text="x")

    elif data.shape[1] == 2:
        fig.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1]))
        if unit:
            fig.update_xaxes(title_text="x ({})".format(unit))
            fig.update_yaxes(title_text="y ({})".format(unit))
        else:
            fig.update_xaxes(title_text="x")
            fig.update_yaxes(title_text="y")
        fig.update_layout(height=600, width=600)

    elif data.shape[1] == 3:
        fig.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2]))

        if unit:
            fig.update_xaxes(title_text="x ({})".format(unit))
            fig.update_yaxes(title_text="y ({})".format(unit))
            # fig.update_zaxes(title_text='z ({})'.format(unit))
        else:
            fig.update_xaxes(title_text="x")
            fig.update_yaxes(title_text="y")
            # fig.update_zaxes(title_text='z')

    fig.update_layout(
        hovermode=False, margin=dict(t=5), yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    return fig



def trial_align_spatial_series(spatial_series, trials=None):
    options = [("x", 0), ("y", 1), ("z", 2)][: spatial_series.data.shape[1]]
    if spatial_series.rate is None:
        return AlignMultiTraceTimeSeriesByTrialsVariable(
            time_series=spatial_series,
            trials=trials,
            trace_controller_kwargs=dict(options=options),
        )
    else:
        return AlignMultiTraceTimeSeriesByTrialsConstant(
            time_series=spatial_series,
            trials=trials,
            trace_controller_kwargs=dict(options=options),
        )
