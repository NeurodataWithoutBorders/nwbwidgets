import numpy as np
from pynwb.core import DynamicTable
from hdmf.common.table import VectorData
from nwbwidgets.utils.dynamictable import infer_categorical_columns



def test_infer_categorical_columns():
    
    data1 = np.array([ True, False,  True,  True,  True,  True, False, False,  True])
    data2 = np.array([False, False,  True,  True, False,  True, False,  True, False])
    
    vd1 = VectorData('spikes','vector data for creating a DynamicTable',data=data1)
    vd2 = VectorData('LFP','vector data for creating a DynamicTable',data=data2)
    vd=[vd1,vd2]
    
    dynamic_table = DynamicTable(name='test table',description='This is a test table',
                                 columns=vd,colnames=['spikes','LFP'])
    
    assert(infer_categorical_columns(dynamic_table)['spikes'].shape[0]==2)
    assert(infer_categorical_columns(dynamic_table)['LFP'].shape[0]==2)
