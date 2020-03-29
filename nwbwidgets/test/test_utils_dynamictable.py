import numpy as np
from pynwb.core import DynamicTable
from hdmf.common.table import VectorData
from nwbwidgets.utils.dynamictable import infer_categorical_columns



def test_infer_categorical_columns():
    
    data1 = np.array([1,2,2,3,1,1,3,2,3])
    data2 = np.array([3,4,2,4,3,2,2,4,4])
    
    vd1 = VectorData('Data1','vector data for creating a DynamicTable',data=data1)
    vd2 = VectorData('Data2','vector data for creating a DynamicTable',data=data2)
    vd=[vd1,vd2]
    
    dynamic_table = DynamicTable(name='test table',description='This is a test table',
                             columns=vd,colnames=['Data1','Data2'])
    
    assert all(infer_categorical_columns(dynamic_table))==all({'Data1': np.array([1, 2, 3]), 'Data2': np.array([2, 3, 4])})
