from nwbwidgets import view
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython import display
from collections import Iterable
from pynwb import TimeSeries
from pynwb.core import DynamicTable, NWBBaseType
from collections import OrderedDict


def show_timeseries(node: TimeSeries, **kwargs):
    info = []
    for key in ('description', 'comments', 'unit', 'resolution', 'conversion'):
        info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    children = [widgets.VBox(info)]

    fig, ax = plt.subplots()
    if node.timestamps:
        ax.plot(node.timestamps, node.data)
    else:
        ax.plot(np.arange(len(node.data)) / node.rate + node.starting_time, node.data)
    ax.set_xlabel('time (s)')
    if node.unit:
        ax.set_ylabel(node.unit)

    children.append(view.fig2widget(fig))

    return widgets.HBox(children=children)


def show_dynamic_table(node: DynamicTable, **kwargs):
    out1 = widgets.Output()
    with out1:
        display.display(node.to_dataframe())
    return out1


def show_neurodata_base(node: NWBBaseType, neurodata_vis_spec: OrderedDict):
    info = []
    neuro_data = []
    labels = []
    for key, value in node.fields.items():
        if isinstance(value, str):
            info.append(widgets.Text(value=repr(value), description=key, disabled=True))
        elif (isinstance(value, Iterable) and len(value)) or value:
            neuro_data.append(view.nwb2widget(value, neurodata_vis_spec=neurodata_vis_spec))
            labels.append(key)
    accordion = widgets.Accordion(children=neuro_data, selected_index=None)
    for i, label in enumerate(labels):
        if hasattr(node.fields[label], 'description') and node.fields[label].description:
            accordion.set_title(i, label + ': ' + node.fields[label].description)
        else:
            accordion.set_title(i, label)
    return widgets.VBox(info + [accordion])
