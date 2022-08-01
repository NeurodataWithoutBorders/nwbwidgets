import unittest

import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import widgets
import plotly.graph_objects as go

from pynwb import TimeSeries
from pynwb.epoch import TimeIntervals
from pynwb.behavior import SpatialSeries

from nwbwidgets.timeseries import (
    BaseGroupedTraceWidget,
    show_ts_fields,
    show_timeseries,
    plot_traces,
    show_indexed_timeseries_mpl,
    AlignMultiTraceTimeSeriesByTrialsConstant,
    AlignMultiTraceTimeSeriesByTrialsVariable,
    SingleTracePlotlyWidget,
    SeparateTracesPlotlyWidget,
    get_timeseries_tt,
    show_indexed_timeseries_plotly,
    show_timeseries_mpl,
)


def test_timeseries_widget():
    ts = TimeSeries(
        name="name",
        description="no description",
        data=np.array([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0]]),
        rate=100.0,
        unit="m",
    )

    BaseGroupedTraceWidget(ts)


def test_show_timeseries_mpl():

    ts = TimeSeries(
        name="name",
        description="no description",
        data=np.array([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0]]),
        rate=100.0,
        unit="m",
    )

    show_timeseries_mpl(ts)
    show_timeseries_mpl(ts, time_window=(0.0, 1.0))


class TestTracesPlotlyWidget(unittest.TestCase):
    def setUp(self):
        data = np.random.rand(160, 3)
        self.ts_multi = SpatialSeries(
            name="test_timeseries",
            data=data,
            reference_frame="lowerleft",
            starting_time=0.0,
            rate=1.0,
        )
        self.ts_single = TimeSeries(
            name="test_timeseries",
            data=data[:, 0],
            unit="m",
            starting_time=0.0,
            rate=1.0,
        )
        # Create 'intermittent' timeseries by defining timestamps with a gap from [10, 20]
        self.ts_intermittent = TimeSeries(
            name="test_timeseries_intermittent",
            data=data[:, 0],
            timestamps=np.hstack([np.arange(0, 10, 10 / data.shape[0] * 2),
                                  np.arange(20, 30, 10 / data.shape[0] * 2)]),
            unit='m',
        )

    def test_single_trace_widget(self):
        single_wd = SingleTracePlotlyWidget(timeseries=self.ts_single)
        tt = get_timeseries_tt(self.ts_single)
        single_wd.controls["time_window"].value = [
            tt[int(len(tt) * 0.2)],
            tt[int(len(tt) * 0.4)],
        ]

    def test_single_trace_widget_null_data(self):
        single_wd = SingleTracePlotlyWidget(timeseries=self.ts_intermittent)
        tt = get_timeseries_tt(self.ts_single, 12, 18)
        single_wd.controls["time_window"].value = [
            tt[int(len(tt) * 0.2)],
            tt[int(len(tt) * 0.4)],
        ]

    def test_separate_traces_widget(self):
        single_wd = SeparateTracesPlotlyWidget(timeseries=self.ts_multi)
        tt = get_timeseries_tt(self.ts_multi)
        single_wd.controls["time_window"].value = [
            tt[int(len(tt) * 0.2)],
            tt[int(len(tt) * 0.4)],
        ]


class TestIndexTimeSeriesPlotly(unittest.TestCase):
    def setUp(self):
        data = np.random.rand(160, 3)
        self.ts = TimeSeries(
            name="name",
            description="no description",
            data=data,
            starting_time=0.0,
            rate=100.0,
            unit="m",
        )
        self.ts_single = TimeSeries(
            name="test_timeseries",
            data=data[:, 0],
            unit="m",
            starting_time=0.0,
            rate=100.0,
        )
        self.tt = get_timeseries_tt(self.ts)

    def test_no_args(self):
        fig_out = show_indexed_timeseries_plotly(timeseries=self.ts)
        fig_out_single = show_indexed_timeseries_plotly(timeseries=self.ts_single)
        assert isinstance(fig_out, go.FigureWidget)
        assert isinstance(fig_out_single, go.FigureWidget)
        assert len(fig_out.data) == 3
        assert len(fig_out_single.data) == 1
        assert np.allclose(fig_out.data[0].x, self.tt)
        assert np.allclose(fig_out_single.data[0].x, self.tt)
        assert np.allclose(fig_out.data[0].y, self.ts.data[:, 0])
        assert np.allclose(fig_out.data[1].y, self.ts.data[:, 1])
        assert np.allclose(fig_out.data[2].y, self.ts.data[:, 2])

    def test_value_errors(self):
        time_window = [
            self.tt[int(len(self.tt) * 0.2)],
            self.tt[int(len(self.tt) * 0.4)],
        ]
        self.assertRaises(
            ValueError,
            show_indexed_timeseries_plotly,
            timeseries=self.ts_single,
            istart=3,
            time_window=time_window,
        )
        self.assertRaises(
            ValueError,
            show_indexed_timeseries_plotly,
            timeseries=self.ts_single,
            trace_range=[2, 5],
        )


class ShowTimeSeriesTestCase(unittest.TestCase):
    def setUp(self):
        data = np.random.rand(160, 3)
        self.ts = TimeSeries(
            name="test_timeseries", data=data, unit="m", starting_time=0.0, rate=1.0
        )

    def test_show_ts_fields(self):
        assert isinstance(show_ts_fields(self.ts), widgets.Widget)

    def test_show_timeseries(self):
        assert isinstance(show_timeseries(self.ts, istart=5, istop=56), widgets.Widget)

    def test_show_indexed_timeseries_mpl(self):
        ax = show_indexed_timeseries_mpl(
            self.ts, zero_start=True, title="Test show_indexed_timeseries_mpl"
        )
        assert isinstance(ax, plt.Subplot)


class PlotTracesTestCase(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(160, 3)

    def test_plot_traces(self):
        ts = TimeSeries(
            name="test_timeseries",
            data=self.data,
            unit="m",
            starting_time=0.0,
            rate=20.0,
        )
        plot_traces(ts)

    def test_plot_traces_fix(self):
        ts = TimeSeries(
            name="test_timeseries",
            data=self.data.T,
            unit="m",
            starting_time=0.0,
            rate=20.0,
        )
        plot_traces(ts)


class TestAlignMultiTraceTimeSeriesByTrials(unittest.TestCase):
    def setUp(self):
        data = np.random.rand(100, 10)
        timestamps = [0.0]
        for _ in range(data.shape[0]):
            timestamps.append(timestamps[-1] + 0.75 + 0.25 * np.random.rand())
        self.ts_rate = TimeSeries(
            name="test_timeseries_rate",
            data=data,
            unit="m",
            starting_time=0.0,
            rate=1.0,
        )
        self.ts_timestamps = TimeSeries(
            name="test_timeseries_timestamps",
            data=data,
            unit="m",
            timestamps=np.array(timestamps),
        )
        self.time_intervals = TimeIntervals(name="Test Time Interval")
        n_intervals = 10
        for start_time in np.linspace(0, 75, n_intervals + 1):
            if start_time < 75:
                stt = start_time + np.random.rand()
                spt = stt + 7 - np.random.rand()
                self.time_intervals.add_interval(start_time=stt, stop_time=spt)
        self.time_intervals.add_column(
            name="temp", description="desc", data=np.random.randint(2, size=n_intervals)
        )
        self.time_intervals.add_column(
            name="temp2",
            description="desc",
            data=np.random.randint(10, size=n_intervals),
        )

    def test_align_by_timestamps(self):
        amt = AlignMultiTraceTimeSeriesByTrialsVariable(
            time_series=self.ts_timestamps, trials=self.time_intervals
        )
        gas = amt.controls['gas']
        gas.group_dd.value = list(gas.categorical_columns.keys())[0]
        gas.group_sm.value = (gas.group_sm.options[0],)
        fig = amt.children[-1]
        assert len(fig.data)==len(gas.group_sm.value)

    def test_align_by_rate(self):
        amt = AlignMultiTraceTimeSeriesByTrialsConstant(
            time_series=self.ts_rate, trials=self.time_intervals
        )
        gas = amt.controls['gas']
        gas.group_dd.value = list(gas.categorical_columns)[0]
        gas.group_sm.value = (gas.group_sm.options[0],)
        fig = amt.children[-1]
        assert len(fig.data) == len(gas.group_sm.value)
