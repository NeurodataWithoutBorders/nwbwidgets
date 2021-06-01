import unittest
from datetime import datetime

import numpy as np
from dateutil.tz import tzlocal
from nwbwidgets.utils.timeseries import (
    get_timeseries_tt,
    get_timeseries_maxt,
    get_timeseries_mint,
    get_timeseries_in_units,
    timeseries_time_to_ind,
    bisect_timeseries_by_times,
    align_by_trials,
    align_by_time_intervals,
)
from pynwb import NWBFile
from pynwb import TimeSeries
from pynwb.epoch import TimeIntervals


def test_get_timeseries_tt():
    data = list(range(100, 200, 10))
    ts = TimeSeries(
        name="test_timeseries", data=data, unit="m", starting_time=0.0, rate=1.0
    )

    tt = get_timeseries_tt(ts)
    np.testing.assert_array_equal(
        tt, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    )


def test_get_timeseries_tt_infstarting_time():
    data = list(range(100, 200, 10))
    ts = TimeSeries(
        name="test_timeseries", data=data, unit="m", starting_time=np.inf, rate=1.0
    )

    tt = get_timeseries_tt(ts)
    np.testing.assert_array_equal(
        tt, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    )


def test_get_timeseries_tt_negativeistop():
    data = list(range(100, 200, 10))
    ts = TimeSeries(
        name="test_timeseries", data=data, unit="m", starting_time=0.0, rate=1.0
    )

    tt = get_timeseries_tt(ts, istop=-1)
    np.testing.assert_array_equal(tt, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])


def test_get_timeseries_in_units():
    data = list(range(100, 200, 10))
    timestamps = list(range(10))
    ts = TimeSeries(
        name="test_timeseries",
        data=data,
        unit="m",
        timestamps=timestamps,
        conversion=np.inf,
    )
    data, unit = get_timeseries_in_units(ts)
    assert unit is None
    assert data == [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]


def test_align_by_trials():
    start_time = datetime(2017, 4, 3, 11, tzinfo=tzlocal())
    create_date = datetime(2017, 4, 15, 12, tzinfo=tzlocal())

    nwbfile = NWBFile(
        session_description="NWBFile for PSTH",
        identifier="NWB123",
        session_start_time=start_time,
        file_create_date=create_date,
    )

    data = np.arange(100, 200, 10)
    timestamps = list(range(10))
    ts = TimeSeries(name="test_timeseries", data=data, unit="m", timestamps=timestamps)
    nwbfile.add_acquisition(ts)

    nwbfile.add_trial_column(
        name="stim", description="the visual stimuli during the trial"
    )
    nwbfile.add_trial(start_time=0.0, stop_time=2.0, stim="person")
    nwbfile.add_trial(start_time=3.0, stop_time=5.0, stim="ocean")
    nwbfile.add_trial(start_time=6.0, stop_time=8.0, stim="desert")

    np.testing.assert_array_equal(align_by_trials(ts), np.array([[110], [140], [170]]))


class TimeSeriesTimeStampTestCase(unittest.TestCase):
    def setUp(self):
        data = np.arange(100, 200, 10)

        timestamps = list(range(1, 4)) + list(range(7, 10)) + list(range(17, 21))
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

    def test_get_timeseries_maxt(self):
        assert get_timeseries_maxt(self.ts_rate) == 9
        assert get_timeseries_maxt(self.ts_timestamps) == 20

    def test_get_timeseries_mint(self):
        assert get_timeseries_mint(self.ts_rate) == 0
        assert get_timeseries_mint(self.ts_timestamps) == 1

    def test_timeseries_time_to_ind(self):
        assert timeseries_time_to_ind(self.ts_rate, 3.3) == 4
        assert timeseries_time_to_ind(self.ts_rate, 15.5) == 9
        assert timeseries_time_to_ind(self.ts_timestamps, 6.5) == 3
        assert timeseries_time_to_ind(self.ts_timestamps, 7.7) == 4
        assert timeseries_time_to_ind(self.ts_timestamps, 27.7) == 9

    def test_bisect_timeseries_by_times(self):
        assert np.array_equal(
            bisect_timeseries_by_times(self.ts_rate, [0, 1, 2], 4),
            [[100, 110, 120, 130], [110, 120, 130, 140], [120, 130, 140, 150]],
        )
        assert isinstance(
            bisect_timeseries_by_times(self.ts_timestamps, [0, 1, 2], 4), list
        )

    def test_get_timeseries_tt_timestamp(self):
        tt = get_timeseries_tt(self.ts_rate)
        np.testing.assert_array_equal(
            tt, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        )

    def test_align_by_time_intervals(self):
        intervals = TimeIntervals(name="Time Intervals")
        np.testing.assert_array_equal(
            align_by_time_intervals(timeseries=self.ts_rate, intervals=intervals),
            np.array([]),
        )
