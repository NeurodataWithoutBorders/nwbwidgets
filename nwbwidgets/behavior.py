from nwbwidgets import view
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from pynwb.behavior import Position, SpatialSeries, BehavioralEvents
from nwbwidgets import base


def show_position(node: Position, neurodata_vis_spec):

    if len(node.spatial_series.keys()) == 1:
        for value in node.spatial_series.values():
            return view.nwb2widget(value, neurodata_vis_spec=neurodata_vis_spec)
    else:
        return view.nwb2widget(node.spatial_series, neurodata_vis_spec=neurodata_vis_spec)


def show_behavioral_events(beh_events: BehavioralEvents, neurodata_vis_spec):
    return base.dict2accordion(beh_events.time_series, neurodata_vis_spec, ls='.')


def show_spatial_series_over_time(node: SpatialSeries, **kwargs):

    text_widget = base.show_text_fields(
        node, exclude=('timestamps_unit', 'comments', 'data', 'timestamps', 'interval'))

    if node.conversion and np.isfinite(node.conversion):
        data = node.data * node.conversion
        unit = node.unit
    else:
        data = node.data
        unit = None

    if len(data.shape) == 1:
        ndims = 1
    else:
        ndims = data.shape[1]

    if ndims == 1:
        fig, ax = plt.subplots()
        if node.timestamps:
            ax.plot(node.timestamps, data, **kwargs)
        else:
            ax.plot(np.arange(len(data)) / node.rate, data, **kwargs)
        ax.set_xlabel('t (sec)')
        if unit:
            ax.set_xlabel('x ({})'.format(unit))
        else:
            ax.set_xlabel('x')
        ax.set_ylabel('x')

    else:
        fig, axs = plt.subplots(ndims, 1, sharex=True)

        for i, (ax, dim_label) in enumerate(zip(axs, ('x', 'y', 'z'))):
            if node.timestamps:
                tt = node.timestamps
            else:
                tt = np.arange(len(data)) / node.rate
            ax.plot(tt, data[:, i], **kwargs)
            if unit:
                ax.set_ylabel(dim_label + ' ({})'.format(unit))
            else:
                ax.set_ylabel(dim_label)
        ax.set_xlabel('t (sec)')

    return widgets.HBox([text_widget, base.fig2widget(fig)])


def show_spatial_series(node: SpatialSeries, **kwargs):

    if node.conversion and np.isfinite(node.conversion):
        data = node.data * node.conversion
        unit = node.unit
    else:
        data = node.data
        unit = None

    if data.shape[0] == 1:
        fig, ax = plt.subplots()
        if node.timestamps:
            ax.plot(node.timestamps, data, **kwargs)
        else:
            ax.plot(np.arange(len(data)) / node.rate, data, **kwargs)
        ax.set_xlabel('t (sec)')
        if unit:
            ax.set_xlabel('x ({})'.format(unit))
        else:
            ax.set_xlabel('x')
        ax.set_ylabel('x')

    elif data.shape[1] == 2:
        fig, ax = plt.subplots()
        ax.plot(data[:, 0], data[:, 1], **kwargs)
        if unit:
            ax.set_xlabel('x ({})'.format(unit))
            ax.set_ylabel('y ({})'.format(unit))
        else:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        ax.axis('equal')

    elif data.shape[1] == 3:
        import ipyvolume.pylab as p3

        fig = p3.figure()
        p3.scatter(data[:, 0], data[:, 1], data[:, 2], **kwargs)
        p3.xlim(np.min(data[:, 0]), np.max(data[:, 0]))
        p3.ylim(np.min(data[:, 1]), np.max(data[:, 1]))
        p3.zlim(np.min(data[:, 2]), np.max(data[:, 2]))

    else:
        raise NotImplementedError

    return fig
