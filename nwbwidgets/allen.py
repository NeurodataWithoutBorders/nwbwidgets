from typing import Iterable

import ipywidgets as widgets
import numpy as np
from hdmf.common import DynamicTable
from pynwb.misc import Units

from .base import lazy_tabs, render_dataframe, TimeIntervalsSelector
from .controllers import GroupAndSortController
from .misc import RasterWidget, PSTHWidget, RasterGridWidget, TuningCurveWidget
from .utils.pynwb import robust_unique
from .view import default_neurodata_vis_spec


class AllenRasterWidget(RasterWidget):
    def make_group_and_sort(self, group_by=None, control_order=False):
        return AllenRasterGroupAndSortController(
            self.units, group_by=group_by, control_order=control_order
        )


class AllenRasterGridWidget(TimeIntervalsSelector):
    InnerWidget = RasterGridWidget


class AllenTuningCurveWidget(TimeIntervalsSelector):
    InnerWidget = TuningCurveWidget


class AllenRasterGroupAndSortController(GroupAndSortController):
    def get_groups(self):

        self.electrodes = self.dynamic_table.get_ancestor("NWBFile").electrodes

        groups = super().get_groups()
        for name in self.electrodes.colnames:
            if not name == "group":
                groups.update(**{name: np.unique(self.electrodes[name][:])})
        return groups

    def get_orderable_cols(self):
        units_orderable_cols = super().get_orderable_cols()
        candidate_cols = [
            x
            for x in self.electrodes.colnames
            if not (
                isinstance(self.electrodes[x][0], Iterable)
                or isinstance(self.electrodes[x][0], str)
            )
        ]
        return units_orderable_cols + [
            x for x in candidate_cols if len(robust_unique(self.electrodes[x][:])) > 1
        ]

    def get_group_vals(self, by, rows_select=()):
        if by is None:
            return None
        elif by in self.dynamic_table:
            return self.dynamic_table[by][:][rows_select]
        else:
            if self.electrodes is not None and by in self.electrodes:
                ids = self.electrodes.id[:]
                inds = [
                    np.argmax(ids == val)
                    for val in self.dynamic_table["peak_channel_id"][:]
                ]
                return self.electrodes[by][:][inds][rows_select]


def allen_show_dynamic_table(node: DynamicTable, **kwargs) -> widgets.Widget:
    if node.name == "electrodes":
        return allen_show_electrodes(node)
    return render_dataframe(node)


def allen_show_electrodes(node: DynamicTable):
    from ccfwidget import CCFWidget

    return lazy_tabs(dict(table=render_dataframe, CCF=CCFWidget), node)


def load_allen_widgets():
    default_neurodata_vis_spec[Units]["Session Raster"] = AllenRasterWidget
    default_neurodata_vis_spec[Units]["Raster Grid"] = AllenRasterGridWidget
    default_neurodata_vis_spec[Units]["Tuning Curves"] = AllenTuningCurveWidget
    # default_neurodata_vis_spec[DynamicTable] = allen_show_dynamic_table
