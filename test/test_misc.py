import unittest
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from dateutil.tz import tzlocal
from ipywidgets import widgets
from nwbwidgets.misc import (
    show_psth_raster,
    PSTHWidget,
    show_decomposition_traces,
    show_decomposition_series,
    RasterWidget,
    show_session_raster,
    show_annotations,
    RasterGridWidget,
    raster_grid,
)
from pynwb import NWBFile
from pynwb.misc import DecompositionSeries, AnnotationSeries


def test_show_psth():
    data = np.random.random([6, 50])
    assert isinstance(show_psth_raster(data=data, start=0, end=1), plt.Subplot)


def test_show_annotations():
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    annotations = AnnotationSeries(name="test_annotations", timestamps=timestamps)
    show_annotations(annotations)


class ShowPSTHTestCase(unittest.TestCase):
    def setUp(self):
        """
        Trials must exist.
        """
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
            spike_times=[2.2, 3.0, 4.5],
            obs_intervals=[[1, 10]],
            location="CA1",
            quality=0.95,
        )
        self.nwbfile.add_unit(
            spike_times=[2.2, 3.0, 25.0, 26.0],
            obs_intervals=[[1, 10], [20, 30]],
            location="CA3",
            quality=0.85,
        )
        self.nwbfile.add_unit(
            spike_times=[1.2, 2.3, 3.3, 4.5],
            obs_intervals=[[1, 10], [20, 30]],
            location="CA1",
            quality=0.90,
        )

        self.nwbfile.add_trial_column(
            name="stim", description="the visual stimuli during the trial"
        )

        self.nwbfile.add_trial(start_time=0.0, stop_time=1.0, stim="person")
        self.nwbfile.add_trial(start_time=0.1, stop_time=2.0, stim="person")
        self.nwbfile.add_trial(start_time=3.0, stop_time=4.0, stim="ocean")
        self.nwbfile.add_trial(start_time=4.0, stop_time=5.0, stim="ocean")
        self.nwbfile.add_trial(start_time=5.0, stop_time=6.0, stim="desert")
        self.nwbfile.add_trial(start_time=6.0, stop_time=8.0, stim="desert")

    def test_psth_widget(self):
        widget = PSTHWidget(self.nwbfile.units)
        assert isinstance(widget, widgets.Widget)

        widget.psth_type_radio = "gaussian"
        widget.trial_event_controller.value = ("start_time", "stop_time")
        widget.unit_controller.value = 1
        widget.gas.group_dd.value = "stim"
        widget.gas.group_dd.value = None

    def test_multipsth_widget(self):
        psth_widget = PSTHWidget(self.nwbfile.units)
        start_labels = ('start_time', 'stop_time')
        fig = psth_widget.update(index=0, start_labels=start_labels)
        assert len(fig.axes) == 2 * len(start_labels)
        
    def test_raster_widget(self):
        assert isinstance(RasterWidget(self.nwbfile.units), widgets.Widget)

    def test_show_session_raster(self):
        assert isinstance(show_session_raster(self.nwbfile.units), plt.Axes)

    def test_raster_grid_widget(self):
        assert isinstance(RasterGridWidget(self.nwbfile.units), widgets.Widget)

    def test_raster_grid(self):
        trials = self.nwbfile.units.get_ancestor("NWBFile").trials
        assert isinstance(
            raster_grid(
                self.nwbfile.units,
                time_intervals=trials,
                index=0,
                start=-0.5,
                end=20.0,
            ),
            plt.Figure,
        )


class ShowDecompositionTestCase(unittest.TestCase):
    def setUp(self):
        data = np.random.rand(160, 2, 3)

        self.ds = DecompositionSeries(
            name="Test Decomposition", data=data, metric="amplitude", rate=1.0
        )

    def test_show_decomposition_traces(self):
        assert isinstance(show_decomposition_traces(self.ds), widgets.Widget)

    def test_show_decomposition_series(self):
        assert isinstance(show_decomposition_series(self.ds), widgets.Widget)
