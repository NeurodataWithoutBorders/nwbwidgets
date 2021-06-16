import unittest

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import widgets
from nwbwidgets.timeseries import (
    BaseGroupedTraceWidget,
    show_ts_fields,
    show_timeseries,
    plot_traces,
    show_indexed_timeseries_mpl,
    AlignMultiTraceTimeSeriesByTrialsConstant,
    AlignMultiTraceTimeSeriesByTrialsVariable,
)
from pynwb import TimeSeries
from pynwb.epoch import TimeIntervals


def test_timeseries_widget():
    ts = TimeSeries(
        name="name",
        description="no description",
        data=np.array([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0]]),
        rate=100.0,
    )

    BaseGroupedTraceWidget(ts)


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
            timestamps=timestamps,
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
        order = gas.value['order']
        fig = amt.children[-1]
        assert len(fig.data)==len(order)

    def test_align_by_rate(self):
        amt = AlignMultiTraceTimeSeriesByTrialsConstant(
            time_series=self.ts_rate, trials=self.time_intervals
        )
        gas = amt.controls['gas']
        gas.group_dd.value = list(gas.categorical_columns)[0]
        order = gas.value['order']
        fig = amt.children[-1]
        assert len(fig.data) == len(order)
