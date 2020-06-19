import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from scipy.signal import stft
from pynwb.ecephys import LFP, SpikeEventSeries, ElectricalSeries
from .base import fig2widget, nwb2widget
from IPython import display
from .timeseries import BaseGroupedTraceWidget


def show_lfp(node: LFP, neurodata_vis_spec: dict):
    lfp = list(node.electrical_series.values())[0]
    return nwb2widget(lfp, neurodata_vis_spec)


def show_spectrogram(neurodata, channel=0, **kwargs):
    fig, ax = plt.subplots()
    f, t, Zxx = stft(neurodata.data[:, channel], neurodata.rate, nperseg=2*17)
    ax.imshow(np.log(np.abs(Zxx)), aspect='auto', extent=[0, max(t), 0, max(f)], origin='lower')
    ax.set_ylim(0, max(f))
    ax.set_xlabel('time')
    ax.set_ylabel('frequency')
    plt.show(ax)


def ElectrodesWidget(node):
    out1 = widgets.Output()
    with out1:
        display.display(node.to_dataframe())
    return out1


def show_spike_event_series(ses: SpikeEventSeries, **kwargs):
    def control_plot(spk_ind):
        fig, ax = plt.subplots(figsize=(9, 5))
        data = ses.data[spk_ind]
        if nChannels > 1:
            for ch in range(nChannels):
                ax.plot(data[:, ch], color='#d9d9d9')
        else:
            ax.plot(data[:], color='#d9d9d9')
        ax.plot(np.mean(data, axis=1), color='k')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        plt.show()
        return fig2widget(fig)

    if len(ses.data.shape) == 3:
        nChannels = ses.data.shape[2]
    else:
        nChannels = ses.data.shape[1]
    nSpikes = ses.data.shape[0]

    # Controls
    field_lay = widgets.Layout(max_height='40px', max_width='100px',
                               min_height='30px', min_width='70px')
    spk_ind = widgets.BoundedIntText(value=0, min=0, max=nSpikes-1,
                                     layout=field_lay)
    controls = {'spk_ind': spk_ind}
    out_fig = widgets.interactive_output(control_plot, controls)

    # Assemble layout box
    lbl_spk = widgets.Label('Spike ID:', layout=field_lay)
    lbl_nspks0 = widgets.Label('N° spikes:', layout=field_lay)
    lbl_nspks1 = widgets.Label(str(nSpikes), layout=field_lay)
    lbl_nch0 = widgets.Label('N° channels:', layout=field_lay)
    lbl_nch1 = widgets.Label(str(nChannels), layout=field_lay)
    hbox0 = widgets.HBox(children=[lbl_spk, spk_ind])
    vbox0 = widgets.VBox(children=[
        widgets.HBox(children=[lbl_nspks0, lbl_nspks1]),
        widgets.HBox(children=[lbl_nch0, lbl_nch1]),
        hbox0
    ])
    hbox1 = widgets.HBox(children=[vbox0, out_fig])

    return hbox1


class ElectricalSeriesWidget(BaseGroupedTraceWidget):
    def __init__(self, electrical_series: ElectricalSeries, neurodata_vis_spec=None, **kwargs):
        super().__init__(electrical_series, 'electrodes', **kwargs)
