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

    def get_trials(self):
        return self.units.get_ancestor('NWBFile').epochs

    def select_trials(self):
        self.controls['trials_select'] = widgets.Dropdown(options=np.unique(self.trials['stimulus_name'][:]).tolist(),
                                                          label='drifting_gratings',
                                                          description='trial select')
        self.children = list(self.children) + [self.controls['trials_select']]

    def process_controls(self, control_states):
        control_states = super().process_controls(control_states)
        control_states['trials_select'] = self.trials['stimulus_name'][:] == control_states.pop('trials_select')
        return control_states

    @staticmethod
    def get_group_vals(dynamic_table, group_by):
        if group_by is None:
            return None
        elif group_by in dynamic_table:
            return dynamic_table[group_by][:]
        else:
            electrodes = dynamic_table.get_ancestor('NWBFile').electrodes
            if electrodes is not None and group_by in electrodes:
                ids = electrodes.id[:]
                inds = [np.argmax(ids == val) for val in dynamic_table['peak_channel_id'][:]]
                return electrodes[group_by][:][inds]


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

