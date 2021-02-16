from datetime import datetime

import ipywidgets as widgets
from dateutil.tz import tzlocal
from nwbwidgets import nwb2widget
from pynwb import NWBFile


def test_nwbfile():
    nwbfile = NWBFile('my first synthetic recording', 'EXAMPLE_ID', datetime.now(tzlocal()),
                      experimenter='Dr. Matthew Douglass',
                      lab='Vision Neuroscience Laboratory',
                      institution='University of Middle Earth at the Shire',
                      experiment_description='We recorded from two macaque monkeys during memory-guided saccade task',
                      session_id='LONELYMTL')

    assert (isinstance(nwb2widget(nwbfile), widgets.Widget))
