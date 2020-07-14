import ipywidgets as widgets
from pathlib import Path
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import numpy as np


class AllenWidget(widgets.VBox):

    def __init__(self, folder_data):
        super().__init__()
        self.lines_select = False
        self.lsdir = [str(e) for e in Path(folder_data).glob('*.nwb')]
        #self.lsnames = [e.name for e in self.lsdir]
        self.start_app()

    def btn_lines_dealer(self, b):
        # Change spike times button state
        self.lines_select = not self.lines_select
        self.axs[0].cla()
        self.axs[1].cla()
        if 'disable' in self.btn_lines.description.lower():
            self.btn_lines.description = 'Enable spike times'
        else:
            self.btn_lines.description = 'Disable spike times'

        self.plot(b=0)

    def plot(self, b):
        # Plot Electrophysiology
        rate_e = self.nwb.acquisition['filtered_membrane_voltage'].rate
        n_samples_e = self.nwb.acquisition['filtered_membrane_voltage'].data.shape[0]
        xx_e = np.arange(n_samples_e) / rate_e
        self.axs[0].plot(xx_e, 500*self.nwb.acquisition['filtered_membrane_voltage'].data[:])

        # Plot Optophysiology
        rate_o = self.nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'].rate
        n_samples_o = self.nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'].data.shape[0]
        xx_o = np.arange(n_samples_o) / rate_o
        self.axs[1].plot(xx_o, self.nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'].data[:])
        self.axs[1].set_xlabel('Time [s]')

        ymax = max(max(500*self.nwb.acquisition['filtered_membrane_voltage'].data[:]), max(self.nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'].data[:])) + 0.4
        ymin = min(min(500*self.nwb.acquisition['filtered_membrane_voltage'].data[:]), min(self.nwb.processing['ophys'].data_interfaces['fluorescence'].roi_response_series['roi_response_series'].data[:])) - 0.5

        self.axs[0].set_ylim(ymin, ymax)
        self.axs[1].set_ylim(ymin, ymax)

        # Plot spike times
        if self.lines_select:
            for spkt in self.nwb.units['spike_times'][:][0]:
                self.axs[0].plot([spkt, spkt], [ymin, ymax], 'k', linewidth=0.4)
                self.axs[1].plot([spkt, spkt], [ymin, ymax], 'k', linewidth=0.4)

    def handle_select(self, b):

        io = NWBHDF5IO(self.selecter.value[0], mode='r')
        self.nwb = io.read()

        self.btn_lines = widgets.Button(description='Enable spike times', button_style='')
        self.btn_lines.on_click(self.btn_lines_dealer)
        btn_back = widgets.Button(description='Select new file')
        btn_back.on_click(self.start_app)

        header_box = widgets.HBox([btn_back, self.btn_lines])

        self.output_fig = widgets.Output()

        with self.output_fig:
            self.fig, self.axs = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True, figsize=(9, 5))
            self.fig.canvas.toolbar_position = 'right'
            self.fig.canvas.toolbar.collapsed = False
            self.fig.canvas.footer_visible = False
            self.fig.canvas.header_visible = False

        self.plot(b=0)

        self.output_box = widgets.VBox([header_box, self.output_fig])

        # Children
        self.children = [self.output_box]

    def start_app(self, b=0):
        '''Show start page'''

        self.selecter = widgets.SelectMultiple(
            options=self.lsdir,
            value=[self.lsdir[0]],
            description='Files',
            disabled=False
        )

        self.btn_select = widgets.Button(description='Select', button_style='')
        self.btn_select.on_click(self.handle_select)

        hbox = widgets.VBox([self.selecter, self.btn_select])

        self.children = [hbox]
