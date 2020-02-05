import numpy as np
from pynwb import TimeSeries
from nwbwidgets.view import default_neurodata_vis_spec
from nwbwidgets.ecephys import show_lfp
from pynwb.ecephys import LFP, ElectricalSeries
from hdmf.common import DynamicTableRegion



def test_show_lfp():
    
    data = np.random.rand(160,12)
    ts = TimeSeries(name='test_timeseries', data=data, unit='m', starting_time=0.0, rate=1.0)
    
    elec_data = np.array([[0,2,1,3],[4,5,6,7],[8,9,10,11]])
    electrodes = DynamicTableRegion(name='electrodes',data=elec_data,
                                    description='raw lfp')
    
    es = ElectricalSeries(name='random data',data=ts,electrodes=electrodes,rate=1.0)
    lfp = LFP(electrical_series=es, name='sample LFP')
    
    
    # Test show_lfp function in ecephys.py
    show_lfp(lfp, default_neurodata_vis_spec)
