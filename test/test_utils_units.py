import unittest

from datetime import datetime
from dateutil.tz import tzlocal

import numpy as np

import ipywidgets as widgets

from pynwb import NWBFile
from pynwb.epoch import TimeIntervals

from nwbwidgets.utils.units import (
    get_min_spike_time,
    align_by_trials,
    align_by_time_intervals,
)
from nwbwidgets.base import TimeIntervalsSelector
from nwbwidgets.misc import TuningCurveWidget, TuningCurveExtendedWidget


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
        self.nwbfile.add_trial(start_time=8.0, stop_time=12.0, stim="person")
        self.nwbfile.add_trial(start_time=13.0, stop_time=15.0, stim="ocean")
        self.nwbfile.add_trial(start_time=16.0, stop_time=18.0, stim="desert")



class ExtendedTimeIntervalSelector(TimeIntervalsSelector):
    InnerWidget = TuningCurveWidget


class ExtendedTimeIntervalSelectorTestCase(UnitsTrialsTestCase):

    def setUp(self):
        super().setUp()

        # add intervals to nwbfile
        ti1 = TimeIntervals(name='intervals', description='experimental intervals')
        ti1.add_interval(start_time=0.0, stop_time=2.0)
        ti1.add_interval(start_time=2.0, stop_time=4.0)
        ti1.add_interval(start_time=4.0, stop_time=6.0)
        ti1.add_interval(start_time=6.0, stop_time=8.0)
        ti1.add_column(name='var1', data=['a', 'b', 'a', 'b'], description='no description')
        self.nwbfile.add_time_intervals(ti1)

        self.widget = ExtendedTimeIntervalSelector(
            input_data=self.nwbfile.units
        )

    def test_make_widget(self):
        assert isinstance(self.widget, widgets.Widget)

    def test_widget_children(self):
        assert len(self.widget.children) == 2

        for i, c in enumerate(self.widget.children):
            assert isinstance(c, widgets.Widget), f'{i}th child of TuningCurve widget is not a widget'

    def test_make_graph(self):
        # rows controller triggers drawing of graphic
        self.widget.children[1].children[1].value = 'var1'

        for i, c in enumerate(self.widget.children):
            assert isinstance(c, widgets.Widget), f'{i}th child of TuningCurve widget is not a widget'


class TuningCurveTestCase(UnitsTrialsTestCase):

    def setUp(self):
        super().setUp()
        self.widget = TuningCurveWidget(
            units=self.nwbfile.units,
            trials=self.nwbfile.trials
        )
        # rows controller triggers drawing of graphic
        self.widget.children[0].children[1].value = 'stim'

    def test_make_widget(self):
        assert isinstance(self.widget, widgets.Widget)

    def test_widget_children(self):
        assert len(self.widget.children) == 2

        for i, c in enumerate(self.widget.children):
            assert isinstance(c, widgets.Widget), f'{i}th child of TuningCurve widget is not a widget'


class TuningCurveRasterGridCombinedTestCase(UnitsTrialsTestCase):

    def setUp(self):
        super().setUp()
        self.widget = TuningCurveExtendedWidget(
            units=self.nwbfile.units,
            trials=self.nwbfile.trials
        )

    def test_make_widget(self):
        assert isinstance(self.widget, widgets.Widget)

    def test_widget_children(self):
        assert len(self.widget.children) == 3

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
            np.array([-5.8, -5., 17., 18.]),
            np.array([-10.8, -10.,  12.,  13.]),
            np.array([-13.8, -13.,   9.,  10.])
        ]

        at = align_by_trials(
            self.nwbfile.units,
            index=1,
            start=-20.0,
            end=30.0
        )

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
            start=-20.0,
            end=30.0,
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
            start=-20.0,
            end=30.0,
            rows_select=[0, 1],
        )

        compare_to_ati = [
            np.array([-18.8, -18.0, 4.0, 5.0]),
            np.array([-19.8, -19.0, 3.0, 4.0]),
        ]

        np.testing.assert_array_equal(ati, compare_to_ati)
