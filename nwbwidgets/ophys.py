import numpy as np
import matplotlib.pyplot as plt
from pynwb.ophys import RoiResponseSeries, DfOverF, PlaneSegmentation, TwoPhotonSeries, ImageSegmentation
from pynwb.base import NWBDataInterface
from ndx_grayscalevolume import GrayscaleVolume
from .utils.cmaps import linear_transfer_function
import ipywidgets as widgets
from .base import show_neurodata_base, get_timeseries_dur, get_timeseries_tt
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
from bisect import bisect


color_wheel = ['red', 'blue', 'green', 'black', 'magenta', 'yellow']


def show_two_photon_series(indexed_timeseries: TwoPhotonSeries, neurodata_vis_spec: dict):
    output = widgets.Output()

    if len(indexed_timeseries.data.shape) == 3:
        def show_image(index=0):
            fig, ax = plt.subplots(subplot_kw={'xticks': [], 'yticks': []})
            ax.imshow(indexed_timeseries.data[index], cmap='gray')
            output.clear_output(wait=True)
            with output:
                plt.show(fig)
    elif len(indexed_timeseries.data.shape) == 4:
        import ipyvolume.pylab as p3

        def show_image(index=0):
            fig = p3.figure()
            p3.volshow(indexed_timeseries.data[index], tf=linear_transfer_function([0, 0, 0], max_opacity=.3))
            output.clear_output(wait=True)
            with output:
                p3.show()
    else:
        raise NotImplementedError

    def on_index_change(change):
        show_image(change.new)

    slider = widgets.IntSlider(value=0, min=0,
                               max=indexed_timeseries.data.shape[0] - 1,
                               orientation='horizontal')
    slider.observe(on_index_change, names='value')
    show_image()

    return widgets.VBox([output, slider])


def show_df_over_f(df_over_f: DfOverF, neurodata_vis_spec: dict):
    if len(df_over_f.roi_response_series) == 1:
        title, input = list(df_over_f.roi_response_series.items())[0]
        return neurodata_vis_spec[RoiResponseSeries](input, neurodata_vis_spec, title=title)
    else:
        return neurodata_vis_spec[NWBDataInterface](df_over_f, neurodata_vis_spec)


def roi_response_series_widget(node: RoiResponseSeries, neurodata_vis_spec: dict = None,
                               time_window_slider: widgets.IntRangeSlider = None,
                               roi_slider: widgets.FloatRangeSlider = None, **kwargs):
    if time_window_slider is None:
        time_window_slider = widgets.FloatRangeSlider(
            value=[0, 100],
            min=0,
            max=get_timeseries_dur(node),
            step=0.1,
            description='time window',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f')

    if roi_slider is None:
        roi_slider = widgets.IntRangeSlider(
            value=[0, min(30, len(node.rois))],
            min=0,
            max=len(node.rois),
            description='units',
            continuous_update=False,
            orientation='horizontal',
            readout=True)

    controls = {
        'roi_response_series': widgets.fixed(node),
        'time_window': time_window_slider,
        'roi_window': roi_slider,
    }
    controls.update({key: widgets.fixed(val) for key, val in kwargs.items()})

    out_fig = widgets.interactive_output(show_roi_response_series, controls)

    control_widgets = widgets.HBox(children=(time_window_slider, roi_slider))
    vbox = widgets.VBox(children=[control_widgets, out_fig])
    return vbox


def show_roi_response_series(roi_response_series: RoiResponseSeries,
                             neurodata_vis_spec: dict = None,
                             time_window=None, roi_window=None, title: str = None):
    """

    :param roi_response_series: pynwb.ophys.RoiResponseSeries
    :param neurodata_vis_spec: OrderedDict
    :param time_window: int
    :param title: str
    :return: matplotlib.pyplot.Figure
    """
    if time_window is None:
        time_window = [None, None]
    if roi_window is None:
        roi_window = [0, len(roi_response_series.rois)]
    tt = get_timeseries_tt(roi_response_series)
    if time_window[0] is None:
        t_ind_start = 0
    else:
        t_ind_start = bisect(tt, time_window[0])
    if time_window[1] is None:
        t_ind_stop = -1
    else:
        t_ind_stop = bisect(tt, time_window[1], t_ind_start)
    data = roi_response_series.data
    tt = tt[t_ind_start: t_ind_stop]
    if data.shape[1] == len(tt):  # fix of orientation is incorrect
        mini_data = data[roi_window[0]:roi_window[1], t_ind_start:t_ind_stop].T
    else:
        mini_data = data[t_ind_start:t_ind_stop, roi_window[0]:roi_window[1]]

    gap = np.median(np.nanstd(mini_data, axis=0)) * 20
    offsets = np.arange(roi_window[1] - roi_window[0]) * gap

    fig, ax = plt.subplots()
    ax.figure.set_size_inches(12, 6)
    ax.plot(tt, mini_data + offsets)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('traces')
    if np.isfinite(gap):
        ax.set_ylim(-gap, offsets[-1] + gap)
        ax.set_xlim(tt[0], tt[-1])
        ax.set_yticks(offsets)
        ax.set_yticklabels(np.arange(roi_window[0], roi_window[1]))

    if title is not None:
        ax.set_title(title)

    return fig


def show_image_segmentation(img_seg: ImageSegmentation, neurodata_vis_spec: dict):
    if len(img_seg.plane_segmentations) == 1:
        return show_plane_segmentation(next(iter(img_seg.plane_segmentations.values())), neurodata_vis_spec)
    else:
        return show_neurodata_base(ImageSegmentation, neurodata_vis_spec)


def show_plane_segmentation(plane_seg: PlaneSegmentation, neurodata_vis_spec: dict):
    nrois = len(plane_seg)

    if 'voxel_mask' in plane_seg:
        import ipyvolume.pylab as p3

        dims = np.array([max(max(plane_seg['voxel_mask'][i][dim]) for i in range(nrois))
                         for dim in ['x', 'y', 'z']]).astype('int') + 1
        fig = p3.figure()
        for icolor, color in enumerate(color_wheel):
            vol = np.zeros(dims)
            sel = np.arange(icolor, nrois, len(color_wheel))
            for isel in sel:
                dat = plane_seg['voxel_mask'][isel]
                vol[tuple(dat['x'].astype('int')),
                    tuple(dat['y'].astype('int')),
                    tuple(dat['z'].astype('int'))] = 1
            p3.volshow(vol, tf=linear_transfer_function(color, max_opacity=.3))
        return fig
    elif 'image_mask' in plane_seg:
        if 'neuron_type' in plane_seg:
            neuron_types = np.unique(plane_seg['neuron_type'][:])
        data = plane_seg['image_mask'].data
        nUnits = data.shape[0]
        fig = go.FigureWidget()
        aux_leg = []
        for i in range(nUnits):
            if plane_seg['neuron_type'][i] not in aux_leg:
                show_leg = True
                aux_leg.append(plane_seg['neuron_type'][i])
            else:
                show_leg = False
            c = color_wheel[np.where(neuron_types==plane_seg['neuron_type'][i])[0][0]]
            # hover text
            hovertext = '<b>roi_id</b>: '+str(plane_seg['roi_id'][i])
            rois_cols = list(plane_seg.colnames)
            rois_cols.remove('roi_id')
            sec_str = '<br>'.join([col+': '+str(plane_seg[col][i]) for col in rois_cols if isinstance(plane_seg[col][i], (int, float, np.integer, np.float, str))])
            hovertext += '<br>'+sec_str
            # form cell borders
            y, x = np.where(plane_seg['image_mask'][i])
            arr = np.vstack((x, y)).T
            hull = ConvexHull(arr)
            vertices = np.append(hull.vertices, hull.vertices[0])
            fig.add_trace(
                go.Scatter(
                    x=arr[vertices, 0],
                    y=arr[vertices, 1],
                    fill='toself',
                    mode='lines',
                    line_color=c,
                    name=plane_seg['neuron_type'][i],
                    legendgroup=plane_seg['neuron_type'][i],
                    showlegend=show_leg,
                    text=hovertext,
                    hovertext='text',
                    line=dict(width=.5),
                )
            )
            fig.update_layout(
                width=700, height=500,
                margin=go.layout.Margin(l=60, r=60, b=60, t=60, pad=1),
                plot_bgcolor="rgb(245, 245, 245)",
            )
        return fig


def show_grayscale_volume(vol: GrayscaleVolume, neurodata_vis_spec: dict):
    import ipyvolume.pylab as p3

    fig = p3.figure()
    p3.volshow(vol.data, tf=linear_transfer_function([0, 0, 0], max_opacity=.1))
    return fig
