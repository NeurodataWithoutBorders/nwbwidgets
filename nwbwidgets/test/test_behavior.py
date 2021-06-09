import unittest

import numpy as np
from nwbwidgets.behavior import (
    show_behavioral_events,
    show_spatial_series_over_time,
    show_spatial_series,
    trial_align_spatial_series,
)
from nwbwidgets.base import show_multi_container_interface
from nwbwidgets.view import default_neurodata_vis_spec
from pynwb import TimeSeries
from pynwb.behavior import Position, SpatialSeries, BehavioralEvents
from pynwb.epoch import TimeIntervals


class ShowSpatialSeriesTestCase(unittest.TestCase):
    def setUp(self):
        self.spatial_series = SpatialSeries(
            name="position",
            data=np.linspace(0, 1, 20),
            rate=50.0,
            reference_frame="starting gate",
        )

    def test_show_position(self):
        position = Position(spatial_series=self.spatial_series)

        show_multi_container_interface(position, default_neurodata_vis_spec)

    def test_show_spatial_series_over_time(self):
        show_spatial_series_over_time(self.spatial_series)

    def test_show_spatial_series(self):
        show_spatial_series(self.spatial_series)


class ShowSpatialSeriesTwoDTestCase(unittest.TestCase):
    def setUp(self):
        self.spatial_series = SpatialSeries(
            name="position",
            data=np.array([np.linspace(0, 1, 20), np.linspace(0, 1, 20)]).T,
            rate=50.0,
            reference_frame="starting gate",
        )

    def test_show_spatial_series_over_time_twoD(self):
        show_spatial_series_over_time(self.spatial_series)

    def test_show_spatial_series_twoD(self):
        show_spatial_series(self.spatial_series)


class ShowSpatialSeriesThreeDTestCase(unittest.TestCase):
    def setUp(self):
        self.spatial_series = SpatialSeries(
            name="position",
            data=np.array(
                [np.linspace(0, 1, 20), np.linspace(0, 1, 20), np.linspace(0, 1, 20)]
            ).T,
            rate=50.0,
            reference_frame="starting gate",
        )

    def test_show_spatial_series_over_time_threeD(self):
        show_spatial_series_over_time(self.spatial_series)

    def test_show_spatial_series_threeD(self):
        show_spatial_series(self.spatial_series)


class SpatialSeriesTrialsAlign(unittest.TestCase):
    def setUp(self) -> None:
        data = np.random.rand(100, 3)
        timestamps = [0.0]
        for _ in range(data.shape[0]):
            timestamps.append(timestamps[-1] + 0.75 + 0.25 * np.random.rand())
        self.spatial_series_rate = SpatialSeries(
            name="position_rate",
            data=data,
            starting_time=0.0,
            rate=1.0,
            reference_frame="starting gate",
        )
        self.spatial_series_ts = SpatialSeries(
            name="position_ts",
            data=data,
            timestamps=timestamps,
            reference_frame="starting gate",
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

    def test_spatial_series_trials_align_rate(self):
        trial_align_spatial_series(self.spatial_series_rate, self.time_intervals)

    def test_spatial_series_trials_align_ts(self):
        trial_align_spatial_series(self.spatial_series_ts, self.time_intervals)


def test_show_behavioral_events():
    data = np.arange(100, 200, 10)
    ts = TimeSeries(
        name="test_timeseries", data=data, unit="m", starting_time=0.0, rate=1.0
    )

    beh_events = BehavioralEvents(time_series=ts)

    show_behavioral_events(beh_events, default_neurodata_vis_spec)
