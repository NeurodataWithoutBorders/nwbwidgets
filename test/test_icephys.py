from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from dateutil.tz import tzlocal
from ipywidgets import widgets
from ndx_icephys_meta.icephys import IntracellularRecordings, Sweeps
from pynwb import NWBFile
from pynwb.base import TimeSeries
from pynwb.device import Device
from pynwb.icephys import (
    IntracellularElectrode,
    VoltageClampSeries,
    VoltageClampStimulusSeries,
)

from nwbwidgets.icephys import IVCurveWidget, show_single_sweep_sequence


def test_show_single_sweep_sequence():
    device = Device(name="Axon Patch-Clamp")
    electrode = IntracellularElectrode(name="Patch Clamp", device=device, description="whole-cell")

    stimulus_data = np.random.rand(160, 2)
    stimulus = TimeSeries(
        name="test_timeseries",
        data=stimulus_data,
        unit="m",
        starting_time=0.0,
        rate=1.0,
    )
    response_data = np.random.rand(160, 2)
    response = TimeSeries(
        name="test_timeseries",
        data=response_data,
        unit="m",
        starting_time=0.0,
        rate=1.0,
    )

    icr = IntracellularRecordings()
    icr.add_recording(
        electrode=electrode,
        stimulus_start_index=0,
        stimulus_index_count=100,
        stimulus=stimulus,
        response_start_index=0,
        response_index_count=100,
        response=response,
    )

    sweeps_table = Sweeps(intracellular_recordings_table=icr)
    assert isinstance(show_single_sweep_sequence(sweeps_table), plt.Figure)


def test_iv_curve():
    # Create an ICEphysFile
    ex_nwbfile = NWBFile(
        session_description="my first recording", identifier="EXAMPLE_ID", session_start_time=datetime.now(tzlocal())
    )

    # Add a device
    ex_device = ex_nwbfile.create_device(name="Heka ITC-1600")

    # Add an intracellular electrode
    ex_electrode = ex_nwbfile.create_icephys_electrode(
        name="elec0", description="a mock intracellular electrode", device=ex_device
    )

    # Create an ic-ephys stimulus
    ex_stimulus = VoltageClampStimulusSeries(
        name="stimulus", data=[1, 2, 3, 4, 5], starting_time=123.6, rate=10e3, electrode=ex_electrode, gain=0.02
    )

    # Create an ic-response
    ex_response = VoltageClampSeries(
        name="response",
        data=[0.1, 0.2, 0.3, 0.4, 0.5],
        conversion=1e-12,
        resolution=np.nan,
        starting_time=123.6,
        rate=20e3,
        electrode=ex_electrode,
        gain=0.02,
        capacitance_slow=100e-12,
        resistance_comp_correction=70.0,
    )

    # (A) Add an intracellular recording to the file
    ex_ir_index = ex_nwbfile.add_intracellular_recording(
        electrode=ex_electrode, stimulus=ex_stimulus, response=ex_response
    )

    # (B) Add a list of sweeps to the simultaneous recordings table
    ex_sweep_index = ex_nwbfile.add_icephys_simultaneous_recording(
        recordings=[
            ex_ir_index,
        ]
    )

    # (C) Add a list of simultaneous recordings table indices as a sequential recording
    ex_sequence_index = ex_nwbfile.add_icephys_sequential_recording(
        simultaneous_recordings=[
            ex_sweep_index,
        ],
        stimulus_type="square",
    )

    # (D) Add a list of sequential recordings table indices as a repetition
    run_index = ex_nwbfile.add_icephys_repetition(
        sequential_recordings=[
            ex_sequence_index,
        ]
    )

    # (E) Add a list of repetition table indices as a experimental condition
    ex_nwbfile.add_icephys_experimental_condition(
        repetitions=[
            run_index,
        ]
    )

    # Make widget
    w = IVCurveWidget(sequential_recordings_table=ex_nwbfile.icephys_sequential_recordings)
    assert isinstance(w, widgets.VBox)
    assert isinstance(w.fig, go.FigureWidget)
