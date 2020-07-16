import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from pynwb.ecephys import LFP, SpikeEventSeries, ElectricalSeries
from pynwb.ophys import RoiResponseSeries
from pynwb import TimeSeries
import ipywidgets as widgets
from nwbwidgets.utils.timeseries import get_timeseries_maxt, get_timeseries_mint
from .controllers import StartAndDurationController,  GroupAndSortController
from .utils.widgets import interactive_output
from .timeseries import _prep_timeseries, color_wheel, StartAndDurationController
from .ophys import RoiResponseSeriesWidget


def plot_grouped_traces_allen(time_series: TimeSeries, time_window=None, order=None, ax=None, figsize=(8, 3),
                        group_inds=None, labels=None, colors=color_wheel, show_legend=True, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if order is None:
        order = np.arange(time_series.data.shape[1])

    mini_data, tt, offsets = _prep_timeseries(time_series, time_window, order)

    if group_inds is not None:
        ugroup_inds = np.unique(group_inds)
        handles = []

        for i, ui in enumerate(ugroup_inds):
            color = colors[ugroup_inds[i] % len(colors)]
            lines_handle = ax.plot(tt, mini_data[:, group_inds == ui],
                                   color=color)
            handles.append(lines_handle[0])

        if show_legend:
            ax.legend(handles=handles[::-1], labels=list(labels[ugroup_inds][::-1]), loc='upper left',
                      bbox_to_anchor=(1.01, 1))
    else:
        ax.plot(tt, mini_data, color='k')

    ax.set_xlim((tt[0], tt[-1]))
    ax.set_xlabel('time (s)')

    ymin = min(time_series.data[:])
    ymax = max(time_series.data[:])

    ax.set_ylim(ymin, ymax)

    if isinstance(time_series, ElectricalSeries):
        ax.set_ylabel('Vm (mV)')
    else:
        ax.set_ylabel('Fluorescence (AU)')

    if len(offsets) > 1:
        ax.set_ylim(-offsets[0], offsets[-1] + offsets[0])
    if len(order) <= 30:
        ax.set_yticks(offsets)
        ax.set_yticklabels(order)
    else:
        ax.set_yticks([])


class BaseAllen(widgets.HBox):
    def __init__(self, time_series: TimeSeries, dynamic_table_region_name=None,
                 foreign_time_window_controller: StartAndDurationController = None,
                 foreign_group_and_sort_controller: GroupAndSortController = None,
                 mpl_plotter=plot_grouped_traces_allen, **kwargs):
        """
        Parameters
        ----------
        time_series: TimeSeries
        dynamic_table_region_name: str, optional
        foreign_time_window_controller: StartAndDurationController, optional
        foreign_group_and_sort_controller: GroupAndSortController, optional
        kwargs
        """

        if dynamic_table_region_name is not None and foreign_group_and_sort_controller is not None:
            raise TypeError('You cannot supply both `dynamic_table_region_name` and `foreign_group_and_sort_controller`.')

        super().__init__()
        self.time_series = time_series

        if foreign_time_window_controller is not None:
            self.time_window_controller = foreign_time_window_controller
        else:
            self.tmin = get_timeseries_mint(time_series)
            self.tmax = get_timeseries_maxt(time_series)
            self.time_window_controller = StartAndDurationController(tmin=self.tmin, tmax=self.tmax, start=self.tmin, duration=5)

        self.controls = dict(
            time_series=widgets.fixed(self.time_series),
            time_window=self.time_window_controller
        )
        if foreign_group_and_sort_controller is None:
            if dynamic_table_region_name is not None:
                dynamic_table_region = getattr(time_series, dynamic_table_region_name)
                table = dynamic_table_region.table
                referenced_rows = dynamic_table_region.data
                discard_rows = [x for x in range(len(table)) if x not in referenced_rows]
                self.gas = GroupAndSortController(dynamic_table=table, start_discard_rows=discard_rows)
                self.controls.update(gas=self.gas)
            else:
                self.gas = None
        else:
            pass
            self.gas = foreign_group_and_sort_controller
            self.controls.update(gas=None)

        self.out_fig = interactive_output(mpl_plotter, self.controls)

        if foreign_time_window_controller:
            right_panel = self.out_fig
        else:
            right_panel = widgets.VBox(
                children=[
                    self.time_window_controller,
                    self.out_fig,
                ],
                layout=widgets.Layout(width="100%")
            )

        self.children = [right_panel]

        self.layout = widgets.Layout(width="100%")


class AllenDashboard(widgets.VBox):
    def __init__(self, nwb):
        super().__init__()

        # self.tmin = get_timeseries_mint(time_series)
        # self.tmax = get_timeseries_maxt(time_series)
        self.time_window_controller = StartAndDurationController(
            tmin=0,
            tmax=120,
            start=0,
            duration=5
        )

        # self.electrical = AllenElectrical(electrical_serie)

        self.fluorescence = RoiResponseSeriesWidget(
            roi_response_series=nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'],
            foreign_time_window_controller=self.time_window_controller,
            foreign_group_and_sort_controller=None,
            dynamic_table_region_name=None
        )

        self.output_box = widgets.VBox([self.time_window_controller, self.fluorescence])

        self.children = [self.output_box]


class AllenElectrical(BaseAllen):
    def __init__(self, electrical_series: ElectricalSeries, neurodata_vis_spec=None,
                 foreign_time_window_controller=None, foreign_group_and_sort_controller=None,
                 **kwargs):
        if foreign_group_and_sort_controller is not None:
            table = None
        else:
            table = 'electrodes'
        super().__init__(electrical_series, table,
                         foreign_time_window_controller=foreign_time_window_controller,
                         foreign_group_and_sort_controller=foreign_group_and_sort_controller,
                         **kwargs)


class AllenOphys(BaseAllen):
    def __init__(self, roi_response_series: RoiResponseSeries, neurodata_vis_spec=None, **kwargs):
        super().__init__(roi_response_series, 'rois', **kwargs)
