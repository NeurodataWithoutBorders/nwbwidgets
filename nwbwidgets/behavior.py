from nwbwidgets import view
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets


def show_position(node):

    if len(node.spatial_series.keys()) == 1:
        for value in node.spatial_series.values():
            return view.nwb2widget(value)
    else:
        return view.nwb2widget(node.spatial_series)


def show_spatial_series(node):

    text_wiget = view.show_text_fields(node, exclude=('timestamp_unit',))

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

    return widgets.HBox([text_wiget, view.fig2widget(fig)])
