import numpy as np
import matplotlib.pyplot as plt
from pynwb.icephys import IntracellularElectrode
from pynwb.base import TimeSeries
from ndx_icephys_meta.icephys import SweepSequences,Sweeps,IntracellularRecordings
from nwbwidgets.icephys import show_single_sweep_sequence
from pynwb.device import Device




def test_show_single_sweep_sequence():
    
    device = Device(name='Axon Patch-Clamp')
    electrode = IntracellularElectrode(name='Patch Clamp',device=device, 
                                       description='whole-cell')
    
    stimulus_data = np.random.rand(160,2)
    stimulus = TimeSeries(name='test_timeseries', data=stimulus_data, unit='m', starting_time=0.0, rate=1.0)
    response_data = np.random.rand(160,2)
    response = TimeSeries(name='test_timeseries', data=response_data, unit='m', starting_time=0.0, rate=1.0)
    
    icr = IntracellularRecordings()
    icr.add_recording(electrode=electrode, stimulus_start_index=0, stimulus_index_count=100, 
                      stimulus=stimulus, response_start_index=0, response_index_count=100, 
                      response=response)
    
    sweeps_table = Sweeps(intracellular_recordings_table=icr)
    node = SweepSequences(sweeps_table=sweeps_table)
    assert isinstance(show_single_sweep_sequence(node),plt.Figure)

