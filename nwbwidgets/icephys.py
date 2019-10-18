import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
import itkwidgets
import itk
from scipy.signal import stft
from pynwb.ecephys import LFP
from .base import fig2widget


def show_patchclamp_series(node, **kwargs):
    ntabs = 2
    # Use Rendering... as a placeholder
    children = [widgets.HTML('Rendering...') for _ in range(ntabs)]

    def on_selected_index(change):
        # Click on Traces Tab
        if change.new == 1 and isinstance(change.owner.children[1], widgets.HTML):
            widget_box = show_voltage_traces(node)
            children[1] = widget_box
            change.owner.children = children

    field_lay = widgets.Layout(max_height='40px', max_width='500px',
                               min_height='30px', min_width='150px')
    info = []
    for key, val in node.fields.items():
        lbl_key = widgets.Label(key+':', layout=field_lay)
        lbl_val = widgets.Label(str(val), layout=field_lay)
        info.append(widgets.HBox(children=[lbl_key, lbl_val]))
    children[0] = widgets.VBox(info)

    tab_nest = widgets.Tab()
    tab_nest.children = children
    tab_nest.set_title(0, 'Fields')
    tab_nest.set_title(1, 'Traces')
    tab_nest.observe(on_selected_index, names='selected_index')
    return tab_nest


def show_voltage_traces(node):
    # Produce figure
    def control_plot(x0, x1):
        fig, ax = plt.subplots(figsize=(18, 10))
        data = node.data[x0:x1]*conversion
        xx = np.arange(x0, x1)
        ax.plot(xx, data)
        ax.set_xlabel('Time [ms]', fontsize=20)
        ax.set_ylabel('Amplitude [V]', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.show()
        return fig2widget(fig)

    conversion = node.conversion
    nSamples = node.data.shape[0]
    if node.timestamps is not None:
        tt = node.timestamps[:]
    else:
        tt = np.arange(0, nSamples)/node.rate

    # Controls
    field_lay = widgets.Layout(max_height='40px', max_width='100px',
                               min_height='30px', min_width='70px')
    x0 = widgets.BoundedIntText(value=0, min=0, max=int(1000*tt[-1]),
                                layout=field_lay)
    x1 = widgets.BoundedIntText(value=20000, min=100, max=int(1000*tt[-1]),
                                layout=field_lay)
    controls = {
        'x0': x0,
        'x1': x1,
    }
    out_fig = widgets.interactive_output(control_plot, controls)

    # Assemble layout box
    lbl_x = widgets.Label('Time [ms]:', layout=field_lay)
    hbox0 = widgets.HBox(children=[lbl_x, x0, x1])
    vbox = widgets.VBox(children=[hbox0, out_fig])
    return vbox


# def show_spike_event_series(ses, **kwargs):
#     def control_plot(spk_ind):
#         fig, ax = plt.subplots(figsize=(9, 5))
#         data = ses.data[spk_ind, :, :]
#         for ch in range(nChannels):
#             ax.plot(data[:, ch], color='#d9d9d9')
#         ax.plot(np.mean(data, axis=1), color='k')
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Amplitude')
#         plt.show()
#         return fig2widget(fig)
#
#     nChannels = ses.data.shape[2]
#     nSpikes = ses.data.shape[0]
#
#     # Controls
#     field_lay = widgets.Layout(max_height='40px', max_width='100px',
#                                min_height='30px', min_width='70px')
#     spk_ind = widgets.BoundedIntText(value=0, min=0, max=nSpikes-1,
#                                      layout=field_lay)
#     controls = {'spk_ind': spk_ind}
#     out_fig = widgets.interactive_output(control_plot, controls)
#
#     # Assemble layout box
#     lbl_spk = widgets.Label('Spike ID:', layout=field_lay)
#     lbl_nspks0 = widgets.Label('N° spikes:', layout=field_lay)
#     lbl_nspks1 = widgets.Label(str(nSpikes), layout=field_lay)
#     lbl_nch0 = widgets.Label('N° channels:', layout=field_lay)
#     lbl_nch1 = widgets.Label(str(nChannels), layout=field_lay)
#     hbox0 = widgets.HBox(children=[lbl_spk, spk_ind])
#     vbox0 = widgets.VBox(children=[
#         widgets.HBox(children=[lbl_nspks0, lbl_nspks1]),
#         widgets.HBox(children=[lbl_nch0, lbl_nch1]),
#         hbox0
#     ])
#     hbox1 = widgets.HBox(children=[vbox0, out_fig])
#
#     return hbox1
