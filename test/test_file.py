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
    panel.source_options_radio.value = "local dir"
    panel.source_options_radio.value = "local file"
    panel.source_options_radio.value = "dandi"

    # Choose DANDI set
    panel.source_path_text.value = "000226"
    # Click accept DANDI set button
    panel.source_path_dandi_button.click()
    # Click accept specific file button
    panel.source_file_dandi_button.click()

    assert len(panel.widgets_panel.children) > 0