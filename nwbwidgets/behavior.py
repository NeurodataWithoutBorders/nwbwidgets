from abc import abstractmethod

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from nwbwidgets import base
from plotly import graph_objects as go
from pynwb.behavior import SpatialSeries, BehavioralEvents
from pynwb import TimeSeries

from .utils.timeseries import (
    get_timeseries_tt,
    get_timeseries_in_units,
    timeseries_time_to_ind,
)
from .timeseries import (
    AlignMultiTraceTimeSeriesByTrialsConstant,
    AlignMultiTraceTimeSeriesByTrialsVariable,
    AbstractTraceWidget,
    SingleTracePlotlyWidget,
    SeparateTracesPlotlyWidget,
)

from .base import lazy_tabs

from .controllers import StartAndDurationController


def show_behavioral_events(beh_events: BehavioralEvents, neurodata_vis_spec: dict):
    return base.dict2accordion(
        beh_events.time_series, neurodata_vis_spec, ls="", marker="|"
    )


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


def route_spatial_series(spatial_series, **kwargs):
    if len(spatial_series.data.shape) == 1:
        dict_ = {
            "over time": SingleTracePlotlyWidget,
            "trial aligned": trial_align_spatial_series,
        }
    elif spatial_series.data.shape[1] == 2:
        dict_ = {
            "over time": SeparateTracesPlotlyWidget,
            "trace": SpatialSeriesTraceWidget2D,
            "trial aligned": trial_align_spatial_series,
        }
    elif spatial_series.data.shape[1] == 3:
        dict_ = {
            "over time": SeparateTracesPlotlyWidget,
            "trace": SpatialSeriesTraceWidget3D,
            "trial aligned": trial_align_spatial_series,
        }
    else:
        return widgets.HTML("Unsupported number of dimensions.")
    return lazy_tabs(dict_, spatial_series)


class SpatialSeriesTraceWidget(AbstractTraceWidget):
    @abstractmethod
    def plot_data(self, data, units, tt):
        return

    @abstractmethod
    def update_plot(self, data, tt):
        return

    def __init__(
        self,
        timeseries: TimeSeries,
        foreign_time_window_controller: StartAndDurationController = None,
        **kwargs,
    ):
        super().__init__(
            timeseries=timeseries,
            foreign_time_window_controller=foreign_time_window_controller,
            **kwargs,
        )

    def set_out_fig(self):
        timeseries = self.controls["timeseries"].value
        time_window = self.controls["time_window"].value

        istart = timeseries_time_to_ind(timeseries, time_window[0])
        istop = timeseries_time_to_ind(timeseries, time_window[1])
        data, units = get_timeseries_in_units(timeseries, istart, istop)

        tt = get_timeseries_tt(timeseries, istart, istop)

        self.plot_data(data, units, tt)

        def on_change(change):
            time_window = self.controls["time_window"].value
            istart = timeseries_time_to_ind(timeseries, time_window[0])
            istop = timeseries_time_to_ind(timeseries, time_window[1])
            data, units = get_timeseries_in_units(timeseries, istart, istop)
            tt = get_timeseries_tt(timeseries, istart, istop)
            self.update_plot(data, tt)

        self.controls["time_window"].observe(on_change)


class SpatialSeriesTraceWidget2D(SpatialSeriesTraceWidget):
    def plot_data(self, data, units, tt):
        if units is None:
            units = "no units"
        self.out_fig = go.FigureWidget(
            data=go.Scattergl(
                x=list(data[:, 0]),
                y=list(data[:, 1]),
                mode="markers+lines",
                marker_color=tt,
                marker_colorscale="Viridis",
                marker_colorbar=dict(thickness=20, title="time (s)"),
                marker_size=5,
            )
        )

        self.out_fig.update_layout(
            title=self.timeseries.name,
            xaxis_title=f"x ({units})",
            yaxis_title=f"y ({units})",
        )

    def update_plot(self, data, tt):
        self.out_fig.data[0].x = list(data[:, 0])
        self.out_fig.data[0].y = list(data[:, 1])
        self.out_fig.update_traces(marker_color=list(tt))


class SpatialSeriesTraceWidget3D(SpatialSeriesTraceWidget):
    def plot_data(self, data, units, tt):
        if units is None:
            units = "no units"

        self.out_fig = go.FigureWidget(
            data=go.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                mode="markers+lines",
                marker_color=tt,
                marker_colorscale="Viridis",
                marker_colorbar=dict(thickness=20, title="time (s)"),
                marker_size=5,
            )
        )

        self.out_fig.update_layout(
            scene=dict(
                xaxis_title=f"x ({units})",
                yaxis_title=f"y ({units})",
                zaxis_title=f"x ({units})",
            )
        )

    def update_plot(self, data, tt):
        self.out_fig.data[0].x = list(data[:, 0])
        self.out_fig.data[0].y = list(data[:, 1])
        self.out_fig.data[0].z = list(data[:, 2])
        self.out_fig.update_traces(marker_color=list(tt))


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
