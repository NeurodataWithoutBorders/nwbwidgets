from ndx_spectrum import Spectrum
import warnings
import matplotlib.pyplot as plt
from collections.abc import Iterable
from h5py import Dataset
from ipywidgets import widgets
import numpy as np
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots


def show_spectrum(node, **kwargs):
    if isinstance(node, Spectrum):
        power = _data_to_array(node.power)
        if len(power.shape)==2:
            no_channels = power.shape[1]
        else:
            no_channels = 1
        out = widgets.Output()
        with out:
            widgets.interact(lambda channel_no: sp(node, channel_no), channel_no=(0, no_channels, 1))
            show_inline_matplotlib_plots()
        return out
    else:
        warnings.warn('neurodatatype not of type: Spectrum')


def get_spectrum_figure(spectrum,channel_no):
    if 'power' in spectrum.fields and 'phase' in spectrum.fields:
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].semilogy(np.asarray(spectrum.frequencies),
                    np.asarray(spectrum.power)[:, channel_no])
        axs[0].set_ylabel('Power')
        axs[1].plot(np.asarray(spectrum.frequencies),
                np.asarray(spectrum.phase)[:, channel_no])
        axs[1].set_ylabel('phase')
        return fig
    elif 'power' in spectrum.fields:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlabel('frequency')
        ax.semilogy(np.asarray(spectrum.frequencies),
                    np.asarray(spectrum.power)[:,channel_no])
        ax.set_ylabel('Power')
        return fig
    elif 'phase' in spectrum.fields:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(np.asarray(spectrum.frequencies),
                np.asarray(spectrum.phase)[:,channel_no])
        ax.set_ylabel('phase')
        ax.set_xlabel('frequency')
        return fig
