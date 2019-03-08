from pynwb import NWBHDF5IO
from scipy.signal import stft
import matplotlib.pyplot as plt
import numpy as np
import itkwidgets
import ipywidgets as widgets
from ipywidgets import Layout
import itk
import pynwb


def _widget_for_nwb_type(node):
    if isinstance(node, pynwb.ecephys.LFP):
        lfp = node.electrical_series['lfp']
        children = [widgets.HTML('Rendering...'), widgets.HTML('Rendering...')]
        def on_selected_index(change):
            if change.new == 1 and isinstance(change.owner.children[1], widgets.HTML):
                slider = widgets.IntSlider(value=0, min=0, max=lfp.data.shape[1]-1, description='Channel', orientation='horizontal')
                def create_spectrogram(channel=0):
                    f,t,Zxx = stft(lfp.data[:,channel],lfp.rate, nperseg=128)
                    spect = np.log(np.abs(Zxx))
                    image = itk.GetImageFromArray(spect)
                    image.SetSpacing([(f[1]-f[0]), (t[1]-t[0])*1e-1])
                    direction = image.GetDirection()
                    vnl_matrix = direction.GetVnlMatrix()
                    vnl_matrix.set(0, 0, 0.0)
                    vnl_matrix.set(0, 1, -1.0)
                    vnl_matrix.set(1, 0, 1.0)
                    vnl_matrix.set(1, 1, 0.0)
                    return image
                spectrogram = create_spectrogram(0)
                viewer = itkwidgets.view(spectrogram, ui_collapsed=True, select_roi=True, annotations=False)
                spect_vbox_children = [slider, viewer]
                spect_vbox = widgets.VBox(spect_vbox_children)
                children[1] = spect_vbox
                change.owner.children = children
                channel_to_spectrogram = {0: spectrogram}
                def on_change_channel(change):
                    channel = change.new
                    if channel not in channel_to_spectrogram:
                        channel_to_spectrogram[channel] = create_spectrogram(channel)
                    viewer.image = channel_to_spectrogram[channel]
                slider.observe(on_change_channel, names='value')

        vbox = []
        for key, value in lfp.fields.items():
            vbox.append(widgets.Text(value=repr(value), description=key, disabled=True))
        children[0] = widgets.VBox(vbox)

        tab_nest = widgets.Tab()
        # Use Rendering... as a placeholder
        tab_nest.children = children
        tab_nest.set_title(0, 'Fields')
        tab_nest.set_title(1, 'Spectrogram')
        tab_nest.observe(on_selected_index, names='selected_index')
        return tab_nest
    elif isinstance(node, pynwb.core.LabelledDict):
        children = []
        for key in node:
            if isinstance(node[key], pynwb.ecephys.LFP):
                children.append((key, _widget_for_nwb_type(node[key])))
            if isinstance(node[key], pynwb.base.ProcessingModule):
                children.append((key, _widget_for_nwb_type(node[key].data_interfaces)))
        accordion = widgets.Accordion(children=[widget for (key, widget) in children], selected_index=None)
        for ii, (key, widget) in enumerate(children):
            if hasattr(node[key], 'description'):
                accordion.set_title(ii, key + ': ' + node[key].description)
            else:
                accordion.set_title(ii, key)
        return accordion
    else:
        raise TypeError('Unknown nwb type')


def view(nwb):
    # List of objects to display
    vbox = []

    children = {}
    datasets = []

    widget = _widget_for_nwb_type(nwb.modules)
    vbox.append(widget)

    return widgets.VBox(vbox)
