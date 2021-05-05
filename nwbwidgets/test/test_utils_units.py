import unittest
from datetime import datetime

import numpy as np
from dateutil.tz import tzlocal
import ipywidgets as widgets
from nwbwidgets.utils.units import (
    get_min_spike_time,
    align_by_trials,
    align_by_time_intervals,
)
from pynwb import NWBFile
from pynwb.epoch import TimeIntervals

from ..misc import TuningCurveWidget


class UnitsTrialsTestCase(unittest.TestCase):

    def setUp(self):
        start_time = datetime(2017, 4, 3, 11, tzinfo=tzlocal())
        create_date = datetime(2017, 4, 15, 12, tzinfo=tzlocal())

        self.nwbfile = NWBFile(
            session_description="NWBFile for PSTH",
            identifier="NWB123",
            session_start_time=start_time,
            file_create_date=create_date,
        )

        self.nwbfile.add_unit_column("location", "the anatomical location of this unit")
        self.nwbfile.add_unit_column(
            "quality", "the quality for the inference of this unit"
        )

        self.nwbfile.add_unit(
            id=1,
            spike_times=[2.2, 3.0, 4.5],
            obs_intervals=[[1, 10]],
            location="CA1",
            quality=0.95,
        )
        self.nwbfile.add_unit(
            id=2,
            spike_times=[2.2, 3.0, 25.0, 26.0],
            obs_intervals=[[1, 10], [20, 30]],
            location="CA3",
            quality=0.85,
        )
        self.nwbfile.add_unit(
            id=3,
            spike_times=[1.2, 2.3, 3.3, 4.5],
            obs_intervals=[[1, 10], [20, 30]],
            location="CA1",
            quality=0.90,
        )

        self.nwbfile.add_trial_column(
            name="stim", description="the visual stimuli during the trial"
        )

        self.nwbfile.add_trial(start_time=0.0, stop_time=2.0, stim="person")
        self.nwbfile.add_trial(start_time=3.0, stop_time=5.0, stim="ocean")
        self.nwbfile.add_trial(start_time=6.0, stop_time=8.0, stim="desert")


class TuningCurveTestCase(UnitsTrialsTestCase):

    def setUp(self):
        super().setUp()
        self.widget = TuningCurveWidget(
            units=self.nwbfile.units,
            trials=self.nwbfile.trials
        )

    def test_make_widget(self):
        assert isinstance(self.widget, widgets.Widget)

    def test_widget_children(self):
        assert len(self.widget.children) == 6

        for i, c in enumerate(self.widget.children):
            assert isinstance(c, widgets.Widget), f'{i}th child of TuningCurve widget is not a widget'


class ShowPSTHTestCase(UnitsTrialsTestCase):

    def test_get_min_spike_time(self):
        assert get_min_spike_time(self.nwbfile.units) == 1.2

    def test_align_by_trials(self):
        compare_to_at = [
            np.array([2.2, 3.0, 25.0, 26.0]),
            np.array([-0.8, 0.0, 22.0, 23.0]),
            np.array([-3.8, -3.0, 19.0, 20.0]),
        ]

        at = align_by_trials(self.nwbfile.units, index=1, before=20.0, after=30.0)

        np.testing.assert_allclose(at, compare_to_at, rtol=1e-02)

    def test_align_by_time_intervals_Nonetrials_select(self):
        time_intervals = TimeIntervals(name="Test Time Interval")
        time_intervals.add_interval(start_time=21.0, stop_time=28.0)
        time_intervals.add_interval(start_time=22.0, stop_time=26.0)
        time_intervals.add_interval(start_time=22.0, stop_time=28.4)

        ati = align_by_time_intervals(
            self.nwbfile.units,
            index=1,
            intervals=time_intervals,
            stop_label=None,
            before=20.0,
            after=30.0,
        )

        compare_to_ati = [
            np.array([-18.8, -18.0, 4.0, 5.0]),
            np.array([-19.8, -19.0, 3.0, 4.0]),
            np.array([-19.8, -19.0, 3.0, 4.0]),
        ]

        np.testing.assert_array_equal(ati, compare_to_ati)

    def test_align_by_time_intervals(self):
        time_intervals = TimeIntervals(name="Test Time Interval")
        time_intervals.add_interval(start_time=21.0, stop_time=28.0)
        time_intervals.add_interval(start_time=22.0, stop_time=26.0)
        time_intervals.add_interval(start_time=22.0, stop_time=28.4)

        ati = align_by_time_intervals(
            self.nwbfile.units,
            index=1,
            intervals=time_intervals,
            stop_label=None,
            before=20.0,
            after=30.0,
            rows_select=[0, 1],
        )

        compare_to_ati = [
            np.array([-18.8, -18.0, 4.0, 5.0]),
            np.array([-19.8, -19.0, 3.0, 4.0]),
        ]

        np.testing.assert_array_equal(ati, compare_to_ati)
