import numpy as np
from pynwb import TimeSeries
from nwbwidgets.view import default_neurodata_vis_spec
from nwbwidgets.ecephys import show_lfp, show_spectrogram, show_spike_event_series
from pynwb.ecephys import LFP, ElectricalSeries, SpikeEventSeries
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile


def test_show_lfp():
    
    nwbfile = NWBFile('my first synthetic recording', 'EXAMPLE_ID', datetime.now(tzlocal()),
                  experimenter='Dr. Matthew Douglass',
                  lab='Vision Neuroscience Laboratory',
                  institution='University of Middle Earth at the Shire',
                  experiment_description='We recorded from two macaque monkeys during memory-guided saccade task',
                  session_id='LONELYMTL')

    device = nwbfile.create_device(name='trodes_rig123')

    electrode_group = nwbfile.create_electrode_group(name='tetrode1',
                                                     description='an example tetrode',
                                                     location='somewhere in the hippocampus',
                                                     device=device)

    for idx in [1, 2, 3, 4]:
        nwbfile.add_electrode(id=idx,
                              x=1.0, y=2.0, z=3.0,
                              imp=float(-idx),
                              location='CA1', filtering='none',
                              group=electrode_group)

    electrode_table_region = nwbfile.create_electrode_table_region([0, 2], 'the first and third electrodes')


    rate = 10.0
    np.random.seed(1234)
    data_len = 1000
    ephys_data = np.random.rand(data_len * 2).reshape((data_len, 2))
    ephys_timestamps = np.arange(data_len) / rate

    ephys_ts = ElectricalSeries('test_ephys_data',
                                ephys_data,
                                electrode_table_region,
                                timestamps=ephys_timestamps,
                                resolution=0.001,
                                comments="This data was randomly generated with numpy, using 1234 as the seed",
                                description="Random numbers generated with numpy.random.rand")
    nwbfile.add_acquisition(ephys_ts)
    
    lfp = LFP(electrical_series=ephys_ts, name='LFP data')
    
    show_lfp(lfp, default_neurodata_vis_spec)

    
def test_show_spectrogram():
    
    data = np.random.rand(160,12)
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)
    
    channel = 3
    show_spectrogram(ts,channel=channel)
    
    

def test_show_spike_event_series():
    
    nwbfile = NWBFile('my first synthetic recording', 'EXAMPLE_ID', datetime.now(tzlocal()),
                  experimenter='Dr. Matthew Douglass',
                  lab='Vision Neuroscience Laboratory',
                  institution='University of Middle Earth at the Shire',
                  experiment_description='We recorded from two macaque monkeys during memory-guided saccade task',
                  session_id='LONELYMTL')

    device = nwbfile.create_device(name='trodes_rig123')

    electrode_group = nwbfile.create_electrode_group(name='tetrode1',
                                                     description='an example tetrode',
                                                     location='somewhere in the hippocampus',
                                                     device=device)

    for idx in [1, 2, 3, 4]:
        nwbfile.add_electrode(id=idx,
                              x=1.0, y=2.0, z=3.0,
                              imp=float(-idx),
                              location='CA1', filtering='none',
                              group=electrode_group)

    electrode_table_region = nwbfile.create_electrode_table_region([0, 2], 'the first and third electrodes')
    
    
    rate = 10.0
    np.random.seed(1234)
    data_len = 1000
    ephys_data = np.random.rand(data_len * 6).reshape((data_len, 2, 3))
    ephys_timestamps = np.arange(data_len) / rate
    ses = SpikeEventSeries(name='test_ephys_data', 
                           data=ephys_data, 
                           timestamps=ephys_timestamps, 
                           electrodes=electrode_table_region, 
                           resolution=0.001, 
                           comments="This data was randomly generated with numpy, using 1234 as the seed", 
                           description="Random numbers generated with numpy.random.rand")
    nwbfile.add_acquisition(ses)
    
    assert isinstance(show_spike_event_series(ses),widgets.Widget)

