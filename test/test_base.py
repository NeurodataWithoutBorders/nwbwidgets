import unittest
import pytest

from datetime import datetime
from dateutil.tz import tzlocal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ipywidgets import widgets

from pynwb import NWBFile, ProcessingModule, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.core import DynamicTable, NWBData
from pynwb.file import Subject

from nwbwidgets.base import (
    show_neurodata_base,
    processing_module,
    nwb2widget,
    show_text_fields,
    fig2widget,
    vis2widget,
    show_fields,
    df2accordion,
    lazy_show_over_data,
    dataset_to_sheet,
)
from nwbwidgets.view import default_neurodata_vis_spec, show_dynamic_table


def test_show_neurodata_base():
    start_time = datetime(2017, 4, 3, 11, tzinfo=tzlocal())
    create_date = datetime(2017, 4, 15, 12, tzinfo=tzlocal())

    nwbfile = NWBFile(
        session_description="demonstrate NWBFile basics",
        identifier="NWB123",
        session_start_time=start_time,
        file_create_date=create_date,
        related_publications="https://doi.org/10.1088/1741-2552/aaa904",
        experimenter="Dr. Pack",
    )

    assert isinstance(
        show_neurodata_base(nwbfile, default_neurodata_vis_spec), widgets.Widget
    )


def test_show_text_fields():
    data = np.random.rand(160, 3)
    ts = TimeSeries(
        name="test_timeseries", data=data, unit="m", starting_time=0.0, rate=1.0
    )
    assert isinstance(show_text_fields(ts), widgets.Widget)


class ProcessingModuleTestCase(unittest.TestCase):
    def setUp(self):
        spatial_series = SpatialSeries(
            name="position",
            data=np.linspace(0, 1, 20),
            rate=50.0,
            reference_frame="starting gate",
        )
        self.position = Position(spatial_series=spatial_series)

    def test_processing_module(self):
        start_time = datetime(2020, 1, 29, 11, tzinfo=tzlocal())
        nwbfile = NWBFile(
            session_description="Test Session",
            identifier="NWBPM",
            session_start_time=start_time,
        )

        behavior_module = ProcessingModule(
            name="behavior", description="preprocessed behavioral data"
        )
        nwbfile.add_processing_module(behavior_module)

        nwbfile.processing["behavior"].add(self.position)

        processing_module(nwbfile.processing["behavior"], default_neurodata_vis_spec)

    def test_nwb2widget(self):
        nwb2widget(self.position, default_neurodata_vis_spec)


def test_fig2widget():
    data = np.random.rand(160, 3)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(data)

    assert isinstance(fig2widget(fig), widgets.Widget)


class TestVis2widget:
    def test_vis2widget_input_widget(self):
        wg = widgets.IntSlider(
            value=7,
            min=0,
            max=10,
            step=1,
            description="Test:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )

        assert isinstance(vis2widget(wg), widgets.Widget)

    def test_vis2widget_input_figure(self):
        data = np.random.rand(160, 3)

        fig = plt.figure(figsize=(10, 5))
        plt.plot(data)

        assert isinstance(vis2widget(fig), widgets.Widget)

    def test_vis2widget_input_other(self):
        data = np.random.rand(160, 3)
        with pytest.raises(ValueError, match="unsupported vis type"):
            vis2widget(data)


def test_show_subject():
    node = Subject(age="8", sex="m", species="macaque")
    show_fields(node)


def test_show_dynamic_table():
    d = {"col1": [1, 2], "col2": [3, 4]}
    dt = DynamicTable.from_dataframe(
        df=pd.DataFrame(data=d), name="Test Dtable", table_description="no description"
    )
    show_dynamic_table(dt)


def test_df2accordion():
    df = pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
    )

    def func_fig(data):
        fig = plt.figure(figsize=(10, 5))
        plt.plot(data)
        return fig

    df2accordion(df=df, by="a", func=func_fig)


def test_df2accordion_single():
    df = pd.DataFrame(np.array([1]), columns=["a"])

    def func_fig(data):
        fig = plt.figure(figsize=(10, 5))
        plt.plot(data)
        return fig

    df2accordion(df=df, by="a", func=func_fig)


def test_lazy_show_over_data():
    list_ = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def func_fig(data):
        fig = plt.figure(figsize=(10, 5))
        plt.plot(data)
        return fig

    assert isinstance(lazy_show_over_data(list_=list_, func_=func_fig), widgets.Widget)


def test_dataset_1d():
    dset = NWBData(data=np.arange(10), name="1d data")
    dataset_to_sheet(dset.data)


def test_dataset_2d():
    dset = NWBData(data=np.random.randn(10, 10), name="2d data")
    dataset_to_sheet(dset.data)


def test_dataset_3d():
    dset = NWBData(data=np.random.randn(10, 10, 10), name="3d data")
    dataset_to_sheet(dset.data)