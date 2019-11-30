import numpy as np
import matplotlib.pyplot as plt
from pynwb.ophys import RoiResponseSeries, DfOverF, PlaneSegmentation, TwoPhotonSeries, ImageSegmentation
from pynwb.base import NWBDataInterface
from ndx_grayscalevolume import GrayscaleVolume
from .utils.cmaps import linear_transfer_function
import ipywidgets as widgets
from .base import show_neurodata_base
from scipy.spatial import ConvexHull
import plotly.graph_objects as go


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


def show_image_segmentation(img_seg: ImageSegmentation, neurodata_vis_spec: dict):
    if len(img_seg.plane_segmentations) == 1:
        return show_plane_segmentation(next(iter(img_seg.plane_segmentations.values())), neurodata_vis_spec)
    else:
        return show_neurodata_base(ImageSegmentation, neurodata_vis_spec)


def show_plane_segmentation_3d(plane_seg: PlaneSegmentation):
    import ipyvolume.pylab as p3

    nrois = len(plane_seg)

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


def show_plane_segmentation_2d(plane_seg: PlaneSegmentation, color_wheel=color_wheel):
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
        c = color_wheel[np.where(neuron_types == plane_seg['neuron_type'][i])[0][0]]
        # hover text
        hovertext = '<b>roi_id</b>: ' + str(plane_seg['roi_id'][i])
        rois_cols = list(plane_seg.colnames)
        rois_cols.remove('roi_id')
        sec_str = '<br>'.join([col + ': ' + str(plane_seg[col][i]) for col in rois_cols if
                               isinstance(plane_seg[col][i], (int, float, np.integer, np.float, str))])
        hovertext += '<br>' + sec_str
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


def show_plane_segmentation(plane_seg: PlaneSegmentation, neurodata_vis_spec: dict):
    if 'voxel_mask' in plane_seg:
        return show_plane_segmentation_3d(plane_seg)
    elif 'image_mask' in plane_seg:
        return show_plane_segmentation_2d(plane_seg)


def show_grayscale_volume(vol: GrayscaleVolume, neurodata_vis_spec: dict):
    import ipyvolume.pylab as p3

    fig = p3.figure()
    p3.volshow(vol.data, tf=linear_transfer_function([0, 0, 0], max_opacity=.1))
    return fig
