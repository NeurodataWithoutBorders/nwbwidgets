from typing import Iterable

import numpy as np

from pynwb.misc import Units
import ipywidgets as widgets

from .misc import RasterWidget, PSTHWidget, RasterGridWidget
from .view import default_neurodata_vis_spec
from .utils.pynwb import robust_unique
from .controllers import GroupAndSortController


class AllenRasterWidget(RasterWidget):
    def make_group_and_sort(self, group_by=None):
        return AllenRasterGroupAndSortController(self.units, group_by=group_by)


class AllenPSTHWidget(PSTHWidget):

    def __init__(self, units: Units, unit_index=0, unit_controller=None, sigma_in_secs=.05, ntt=1000):

        super().__init__(units, unit_index, unit_controller, sigma_in_secs, ntt)

        self.stimulus_type_dd = widgets.Dropdown(options=np.unique(self.trials['stimulus_name'][:]).tolist(),
                                                 label='drifting_gratings',
                                                 description='stimulus type')

        self.stimulus_type_dd.observe(self.stimulus_type_dd_callback)

        self.children = [self.stimulus_type_dd] + list(self.children)

    def get_trials(self):
        return self.units.get_ancestor('NWBFile').epochs

    def stimulus_type_dd_callback(self, change):
        self.gas.discard_rows = np.where(self.trials['stimulus_name'][:] != self.stimulus_type_dd.value)[0]

    def make_group_and_sort(self, window=False):
        discard_rows = np.where(self.trials['stimulus_name'][:] != 'drifting_gratings')[0]
        gas = GroupAndSortController(self.trials, window=window, start_discard_rows=discard_rows)

        return gas


class AllenRasterGroupAndSortController(GroupAndSortController):

    def get_groups(self):

        self.electrodes = self.dynamic_table.get_ancestor('NWBFile').electrodes

        groups = super().get_groups()
        groups.update({name: np.unique(self.electrodes[name][:]) for name in self.electrodes.colnames})
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


class AllenRasterGridWidget(RasterGridWidget):
    def get_trials(self):
        return self.units.get_ancestor('NWBFile').epochs

    def select_trials(self):
        self.controls['trials_select'] = widgets.Dropdown(options=np.unique(self.trials['stimulus_name'][:]).tolist(),
                                                          label='drifting_gratings',
                                                          description='trial select')
        self.children = list(self.children) + [self.controls['trials_select']]

    def process_controls(self, control_states):
        control_states['trials_select'] = self.trials['stimulus_name'][:] == control_states.pop('trials_select')
        return control_states


def load_allen_widgets():
    default_neurodata_vis_spec[Units]['Session Raster'] = AllenRasterWidget
    default_neurodata_vis_spec[Units]['Grouped PSTH'] = AllenPSTHWidget
    default_neurodata_vis_spec[Units]['Raster Grid'] = AllenRasterGridWidget

