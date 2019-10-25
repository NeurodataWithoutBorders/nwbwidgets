import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from scipy.signal import stft
from pynwb.ecephys import LFP
from .base import fig2widget, nwb2widget


def show_lfp(node: LFP, neurodata_vis_spec):
    lfp = list(node.electrical_series.values())[0]
    return nwb2widget(lfp, neurodata_vis_spec)


def show_voltage_traces(lfp):
    # Produce figure
    def control_plot(x0, x1, ch0, ch1):
        fig, ax = plt.subplots(figsize=(18, 10))
        istart = int(x0 * lfp.rate)
        istop = int(x1 * lfp.rate)
        data = lfp.data[istart:istop, ch0:ch1+1]
        tt = np.linspace(x0, x1, istop-istart)
        mu_array = np.nanmean(data, 0)
        sd_array = np.nanstd(data, 0)
        offset = np.nanmean(sd_array)*5
        yticks = [i*offset for i in range(ch1+1-ch0)]
        for i in range(ch1+1-ch0):
            ax.plot(tt, data[:, i] - mu_array[i] + yticks[i])
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Ch #', fontsize=20)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(i) for i in range(ch0, ch1+1)])
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlim((x0, x1))
        plt.show()
        return fig2widget(fig)

    fs = lfp.rate
    nSamples = lfp.data.shape[0]
    nChannels = lfp.data.shape[1]

    # Controls
    field_lay = widgets.Layout(max_height='40px', max_width='100px',
                               min_height='30px', min_width='70px')
    x0 = widgets.BoundedIntText(value=0, min=0, max=int(1000*nSamples/fs-100),
                                layout=field_lay)
    x1 = widgets.BoundedIntText(value=10, min=0, max=int(1000*nSamples/fs),
                                layout=field_lay)
    ch0 = widgets.BoundedIntText(value=0, min=0, max=int(nChannels-1), layout=field_lay)
    ch1 = widgets.BoundedIntText(value=10, min=0, max=int(nChannels-1), layout=field_lay)

    controls = {
        'x0': x0,
        'x1': x1,
        'ch0': ch0,
        'ch1': ch1
    }
    out_fig = widgets.interactive_output(control_plot, controls)

    # Assemble layout box
    lbl_x = widgets.Label('Time (s):', layout=field_lay)
    lbl_ch = widgets.Label('Ch #:', layout=field_lay)
    lbl_blank = widgets.Label('    ', layout=field_lay)
    hbox0 = widgets.HBox(children=[lbl_x, x0, x1, lbl_blank, lbl_ch, ch0, ch1])
    vbox = widgets.VBox(children=[hbox0, out_fig])
    return vbox


def show_spectrogram(neurodata, channel=0, **kwargs):
    fig, ax = plt.subplots()
    f, t, Zxx = stft(neurodata.data[:, channel], neurodata.rate, nperseg=2*17)
    ax.imshow(np.log(np.abs(Zxx)), aspect='auto', extent=[0, max(t), 0, max(f)], origin='lower')
    ax.set_ylim(0, 50)
    ax.set_xlabel('time')
    ax.set_ylabel('frequency')
    plt.show(ax.figure())


def show_spike_event_series(ses, **kwargs):
    def control_plot(spk_ind):
        fig, ax = plt.subplots(figsize=(9, 5))
        data = ses.data[spk_ind, :, :]
        for ch in range(nChannels):
            ax.plot(data[:, ch], color='#d9d9d9')
        ax.plot(np.mean(data, axis=1), color='k')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        plt.show()
        return fig2widget(fig)

    nChannels = ses.data.shape[2]
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
