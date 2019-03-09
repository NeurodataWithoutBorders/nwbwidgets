from pynwb import NWBHDF5IO
from scipy.signal import stft
import matplotlib.pyplot as plt
import numpy as np
import itkwidgets
import ipywidgets as widgets
from ipywidgets import Layout
import itk
import pynwb
from IPython import display
from collections import Iterable
import time


def mpl_fig2widget(fig):
    out = widgets.Output()
    with out:
        plt.show(fig)
    return out


def show_annotations1(annotations):
    fig, ax = plt.subplots()
    ax.plot(annotations.timestamps, np.ones(len(annotations.timestamps)), '.')
    ax.set_xlabel('time (s)')
    return mpl_fig2widget(fig)


def dict2accordion(d):
    accordion = widgets.Accordion(children=[nwb2widget(x)
                                            for x in d.values()],
                                  selected_index=None)
    for i, label in enumerate(d):
        if hasattr(d[label], 'description') and d[label].description:
            accordion.set_title(i, label + ': ' + d[label].description)
        else:
            accordion.set_title(i, label)
        accordion.set_title(i, label)

    return accordion


def show_text_fields(node):
    info = []
    for key in ('description',):
        info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    return widgets.VBox(info)


def show_dynamic_table(node):
    out1 = widgets.Output()
    with out1:
        display.display(node.to_dataframe())
    return out1


def show_timeseries(node):
    info = []
    for key in ('description', 'comments', 'unit', 'resolution', 'conversion'):
        info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    children = [widgets.VBox(info)]

    out1 = widgets.Output()

    with out1:
        fig, ax = plt.subplots()
        if node.timestamps:
            ax.plot(node.timestamps, node.data)
        else:
            ax.plot(np.arange(len(node.data)) / node.rate, node.data)
        plt.show(fig)

    children.append(out1)

    return widgets.VBox(children=children)


def show_lfp(node):
    lfp = node.electrical_series['lfp']
    ntabs = 3
    children = [widgets.HTML('Rendering...') for _ in range(ntabs)]

    def on_selected_index(change):
        if change.new == 1 and isinstance(change.owner.children[1], widgets.HTML):
            slider = widgets.IntSlider(value=0, min=0, max=lfp.data.shape[1] - 1, description='Channel',
                                       orientation='horizontal')

            def create_spectrogram(channel=0):
                f, t, Zxx = stft(lfp.data[:, channel], lfp.rate, nperseg=128)
                spect = np.log(np.abs(Zxx))
                image = itk.GetImageFromArray(spect)
                image.SetSpacing([(f[1] - f[0]), (t[1] - t[0]) * 1e-1])
                direction = image.GetDirection()
                vnl_matrix = direction.GetVnlMatrix()
                vnl_matrix.set(0, 0, 0.0)
                vnl_matrix.set(0, 1, -1.0)
                vnl_matrix.set(1, 0, 1.0)
                vnl_matrix.set(1, 1, 0.0)
                return image

            spectrogram = create_spectrogram(0)

            viewer = itkwidgets.view(spectrogram, ui_collapsed=True, select_roi=True, annotations=False)
            spect_vbox = widgets.VBox([slider, viewer])
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
    tab_nest.set_title(2, 'test')
    tab_nest.observe(on_selected_index, names='selected_index')
    return tab_nest


def show_neurodata_base(node):
    info = []
    neuro_data = []
    labels = []
    for key, value in node.fields.items():
        if isinstance(value, str):
            info.append(widgets.Text(value=repr(value), description=key, disabled=True))
        elif (isinstance(value, Iterable) and len(value)) or value:
            neuro_data.append(nwb2widget(value))
            labels.append(key)
    accordion = widgets.Accordion(children=neuro_data, selected_index=None)
    for i, label in enumerate(labels):
        if hasattr(node.fields[label], 'description') and node.fields[label].description:
            accordion.set_title(i, label + ': ' + node.fields[label].description)
        else:
            accordion.set_title(i, label)
    return widgets.VBox(info + [accordion])


def show_spatial_series(node):
    info = []
    for key in ('description', 'comments', 'unit', 'resolution', 'conversion'):
        info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    children = [widgets.VBox(info)]

    out1 = widgets.Output()

    with out1:
        if node.conversion and np.isfinite(node.conversion):
            data = node.data * node.conversion
            unit = node.unit
        else:
            data = node.data
            unit = None

        fig, ax = plt.subplots()
        if data.shape[0] == 1:
            if node.timestamps:
                ax.plot(node.timestamps, data)
            else:
                ax.plot(np.arange(len(data)) / node.rate, data)
            ax.set_xlabel('t (sec)')
            if unit:
                ax.set_xlabel('x ({})'.format(unit))
            else:
                ax.set_xlabel('x')
            ax.set_ylabel('x')
        elif data.shape[1] == 2:
            ax.plot(data[:, 0], data[:, 1])
            if unit:
                ax.set_xlabel('x ({})'.format(unit))
                ax.set_ylabel('y ({})'.format(unit))
            else:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
            ax.axis('equal')
        plt.show(fig)

    children.append(out1)

    tab = widgets.Tab(children=children)
    tab.set_title(0, 'info')
    tab.set_title(1, 'trace')

    return tab


def show_position(node):

    if len(node.spatial_series.keys()) == 1:
        for value in node.spatial_series.values():
            return nwb2widget(value)
    else:
        return nwb2widget(node.spatial_series)


neurodata_vis = (
    (
        pynwb.misc.AnnotationSeries, (
            ('text', show_text_fields),
            ('times', show_annotations1))
    ),
    (pynwb.core.LabelledDict, dict2accordion),
    (pynwb.ProcessingModule, lambda x: nwb2widget(x.data_interfaces)),
    (pynwb.core.DynamicTable, show_dynamic_table),
    (pynwb.ecephys.LFP, show_lfp),
    (pynwb.behavior.Position, show_position),
    (pynwb.behavior.SpatialSeries, show_spatial_series),
    (pynwb.TimeSeries, show_timeseries),
    (pynwb.core.NWBBaseType, show_neurodata_base)
)
    

def nwb2widget(node,  neurodata_vis=neurodata_vis):

    for ndtype, widget_funcs in neurodata_vis:
        if isinstance(node, ndtype):
            if isinstance(widget_funcs, tuple):
                children = [widget_funcs[0][1](node)] + [widgets.HTML('Rendering...')
                                                         for _ in range(len(widget_funcs) - 1)]
                tab = widgets.Tab(children=children)
                [tab.set_title(i, label) for i, (label, _) in enumerate(widget_funcs)]

                def on_selected_index(change):
                    if isinstance(change.owner.children[change.new], widgets.HTML):
                        children[change.new] = widget_funcs[change.new][1](node)
                        change.owner.children = children

                tab.observe(on_selected_index, names='selected_index')

                return tab
            else:
                return widget_funcs(node)
    out1 = widgets.Output()
    with out1:
        print(node)
    return out1


def show_spectrogram(neurodata, channel=0):
    fig, ax = plt.subplots()
    f, t, Zxx = stft(neurodata.data[:, channel], neurodata.rate, nperseg=2*17)
    ax.imshow(np.log(np.abs(Zxx)), aspect='auto', extent=[0, max(t), 0, max(f)], origin='lower')
    ax.set_ylim(0, 50)
    plt.show(ax.figure())


def view(nwb):
    # List of objects to display
    vbox = []

    widget = nwb2widget(nwb)
    vbox.append(widget)

    return widgets.VBox(vbox)
