import numpy as np
from hdmf.common.table import VectorData
from nwbwidgets.utils.dynamictable import infer_categorical_columns
from nwbwidgets.utils.testing import dicts_exact_equal
from pynwb.core import DynamicTable
from pynwb.ecephys import ElectrodeGroup
from pynwb.device import Device


def test_infer_categorical_columns():
    data1 = np.array([1, 2, 2, 3, 1, 1, 3, 2, 3])
    data2 = np.array([3, 4, 2, 4, 3, 2, 2, 4, 4])
    device = Device(name='device')
    eg_1 = ElectrodeGroup(name='electrodegroup1',description='desc',location='brain',device=device)
    eg_2 = ElectrodeGroup(name='electrodegroup2', description='desc', location='brain', device=device)
    data3 = [eg_1,eg_2,eg_1,eg_1,eg_1,eg_1,eg_1,eg_1,eg_1]
    vd1 = VectorData("Data1", "vector data for creating a DynamicTable", data=data1)
    vd2 = VectorData("Data2", "vector data for creating a DynamicTable", data=data2)
    vd3 = VectorData("ElectrodeGroup", "vector data for creating a DynamicTable", data=data3)
    vd = [vd1, vd2, vd3]

    dynamic_table = DynamicTable(
        name="test table",
        description="This is a test table",
        columns=vd,
        colnames=["Data1", "Data2", "ElectrodeGroup"],
    )
    print(infer_categorical_columns(dynamic_table))
    assert dicts_exact_equal(
        infer_categorical_columns(dynamic_table),
        {"Data1": data1, "Data2": data2, "ElectrodeGroup": [i.name for i in data3]},
    )
