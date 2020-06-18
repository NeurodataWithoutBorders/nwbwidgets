from typing import Iterable

import numpy as np

from pynwb.misc import Units
import ipywidgets as widgets
from hdmf.common import DynamicTable

from .misc import RasterWidget, PSTHWidget, RasterGridWidget
from .view import default_neurodata_vis_spec
from .utils.pynwb import robust_unique
from .controllers import GroupAndSortController
from .base import lazy_tabs, render_dataframe


class AllenRasterWidget(RasterWidget):
    def make_group_and_sort(self, group_by=None):
        return AllenRasterGroupAndSortController(self.units, group_by=group_by)


class TimeIntervalsSelector(widgets.VBox):

    InnerWidget = None

    def __init__(self, units, **kwargs):

        super().__init__()
        self.units = units
        self.kwargs = kwargs
        self.intervals_tables = units.get_ancestor('NWBFile').intervals
        self.stimulus_type_dd = widgets.Dropdown(options=list(self.intervals_tables.keys()),
                                                 description='stimulus type')
        trials = list(self.intervals_tables.values())[0]
        self.stimulus_type_dd.observe(self.stimulus_type_dd_callback)
        psth_widget = self.InnerWidget(units, trials, **kwargs)
        self.children = [self.stimulus_type_dd, psth_widget]

    def stimulus_type_dd_callback(self, change):
        self.children = [self.stimulus_type_dd, widgets.HTML('Rendering...')]
        trials = self.intervals_tables[self.stimulus_type_dd.value]
        psth_widget = self.InnerWidget(self.units, trials, **self.kwargs)
        self.children = [self.stimulus_type_dd, psth_widget]


class AllenPSTHWidget(TimeIntervalsSelector):
    InnerWidget = PSTHWidget


class AllenRasterGridWidget(TimeIntervalsSelector):
    InnerWidget = RasterGridWidget


class AllenRasterGroupAndSortController(GroupAndSortController):

    def get_groups(self):

        self.electrodes = self.dynamic_table.get_ancestor('NWBFile').electrodes

        groups = super().get_groups()
        for name in self.electrodes.colnames:
            if not name == 'group':
                groups.update(**{name: np.unique(self.electrodes[name][:])})
        return groups

    def get_orderable_cols(self):
        units_orderable_cols = super().get_orderable_cols()
        candidate_cols = [x for x in self.electrodes.colnames
                          if not (isinstance(self.electrodes[x][0], Iterable) or
                                  isinstance(self.electrodes[x][0], str))]
        return units_orderable_cols + [x for x in candidate_cols
                                       if len(robust_unique(self.electrodes[x][:])) > 1]

    def get_group_vals(self, by, rows_select=()):
        if by is None:
            return None
        elif by in self.dynamic_table:
            return self.dynamic_table[by][:][rows_select]
        else:
            if self.electrodes is not None and by in self.electrodes:
                ids = self.electrodes.id[:]
                inds = [np.argmax(ids == val) for val in self.dynamic_table['peak_channel_id'][:]]
                return self.electrodes[by][:][inds][rows_select]

"""
class AllenRasterGridWidget(RasterGridWidget):
    def get_trials(self):
        return self.units.get_ancestor('NWBFile').intervals['dot_motion_presentations']

    def select_trials(self):
        self.intervals_tables = self.units.get_ancestor('NWBFile').intervals

        self.stimulus_type_dd = widgets.Dropdown(options=list(self.intervals_tables.keys()),
                                                 description='stimulus type')

        self.stimulus_type_dd.observe(self.stimulus_type_dd_callback)
        self.children = list(self.children) + [self.stimulus_type_dd]

    def stimulus_type_dd_callback(self, change):
        self.trials = self.intervals_tables[self.stimulus_type_dd.value]
"""


def allen_show_dynamic_table(node: DynamicTable, **kwargs) -> widgets.Widget:
    if node.name == 'electrodes':
        return allen_show_electrodes(node)
    return render_dataframe(node)


def allen_show_electrodes(node: DynamicTable):
    from ccfwidget import CCFWidget

    return lazy_tabs(
        dict(
            table=render_dataframe,
            CCF=CCFWidget
        ),
        node
    )


def load_allen_widgets():
    default_neurodata_vis_spec[Units]['Session Raster'] = AllenRasterWidget
    default_neurodata_vis_spec[Units]['Grouped PSTH'] = AllenPSTHWidget
    default_neurodata_vis_spec[Units]['Raster Grid'] = AllenRasterGridWidget
    default_neurodata_vis_spec[DynamicTable] = allen_show_dynamic_table

