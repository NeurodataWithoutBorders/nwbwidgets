import unittest
import numpy as np
from nwbwidgets.controllers import RangeController, GroupAndSortController
from hdmf.common import DynamicTable, VectorData
from pynwb.ecephys import ElectrodeGroup, Device


class FloatRangeControllerTestCase(unittest.TestCase):
    def setUp(self):
        self.range_controller = RangeController(vmin=0, vmax=10, start_value=(5, 7))

    def test_move_range_slider_down_bigger(self):
        self.range_controller.value = (4, 6)
        self.range_controller.move_down("filler")
        assert self.range_controller.value == (2, 4)

    def test_move_range_slider_down_smaller(self):
        self.range_controller.value = (2, 6)
        self.range_controller.move_down("filler")
        assert self.range_controller.value == (0, 4)

    def test_move_range_slider_up_smaller(self):
        self.range_controller.value = (5, 7)
        self.range_controller.move_up("filler")
        assert self.range_controller.value == (7, 9)

    def test_move_range_slider_up_bigger(self):
        self.range_controller.value = (5, 8)
        self.range_controller.move_up("filler")
        assert self.range_controller.value == (7, 10)


class TestGroupAndSortController(unittest.TestCase):
    def setUp(self) -> None:
        data1 = np.array([1, 2, 2, 3, 1, 1, 3, 2, 3])
        data2 = np.array([3, 4, 2, 4, 3, 2, 2, 4, 4])
        device = Device(name="device")
        eg_1 = ElectrodeGroup(
            name="electrodegroup1", description="desc", location="brain", device=device
        )
        eg_2 = ElectrodeGroup(
            name="electrodegroup2", description="desc", location="brain", device=device
        )
        data3 = [eg_1, eg_2, eg_1, eg_1, eg_1, eg_1, eg_1, eg_1, eg_1]
        vd1 = VectorData("Data1", "vector data for creating a DynamicTable", data=data1)
        vd2 = VectorData("Data2", "vector data for creating a DynamicTable", data=data2)
        vd3 = VectorData(
            "ElectrodeGroup", "vector data for creating a DynamicTable", data=data3
        )
        vd = [vd1, vd2, vd3]

        self.dynamic_table = DynamicTable(
            name="test table",
            description="This is a test table",
            columns=vd,
            colnames=["Data1", "Data2", "ElectrodeGroup"],
        )

    def test_all_rows(self):
        GroupAndSortController(dynamic_table=self.dynamic_table)

    def test_keep_rows(self):
        keep_rows = np.arange(len(self.dynamic_table) // 2)
        GroupAndSortController(dynamic_table=self.dynamic_table, keep_rows=keep_rows)
