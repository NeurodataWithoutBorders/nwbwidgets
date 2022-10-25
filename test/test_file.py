from datetime import datetime
from dateutil.tz import tzlocal

import ipywidgets as widgets

from pynwb import NWBFile

from nwbwidgets import nwb2widget
from nwbwidgets.panel import Panel


def test_nwbfile():
    nwbfile = NWBFile(
        "my first synthetic recording",
        "EXAMPLE_ID",
        datetime.now(tzlocal()),
        experimenter="Dr. Matthew Douglass",
        lab="Vision Neuroscience Laboratory",
        institution="University of Middle Earth at the Shire",
        experiment_description="We recorded from two macaque monkeys during memory-guided saccade task",
        session_id="LONELYMTL",
    )
    assert isinstance(nwb2widget(nwbfile), widgets.Widget)


def test_panel():
    panel = Panel()
    assert isinstance(panel, widgets.Widget)

    # Change dropdown options for coverage
    panel.source_options_radio.value = "Local dir"
    panel.source_options_radio.value = "Local file"
    panel.source_options_radio.value = "S3"
    panel.source_options_radio.value = "DANDI"

    # Choose DANDI set
    panel.source_dandi_id.value = panel.source_dandi_id.value.options[10]
    # Click accept specific file button
    panel.source_dandi_file_button.click()

    assert len(panel.widgets_panel.children) > 0