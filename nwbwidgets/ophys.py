import numpy as np
import matplotlib.pyplot as plt
from pynwb.ophys import RoiResponseSeries, DfOverF, PlaneSegmentation, TwoPhotonSeries, ImageSegmentation
from pynwb.base import NWBDataInterface
from ndx_grayscalevolume import GrayscaleVolume
from .utils.cmaps import linear_transfer_function
from .utils.dynamictable import infer_categorical_columns
from .utils.functional import MemoizeMutable
import ipywidgets as widgets
import plotly.graph_objects as go
from skimage import measure

from .timeseries import BaseGroupedTraceWidget



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
            p3.figure()
            p3.volshow(indexed_timeseries.data[index], tf=linear_transfer_function([0, 0, 0], max_opacity=.3))
            output.clear_output(wait=True)
            with output:
                p3.show()
    else:
        raise NotImplementedError

    slider = widgets.IntSlider(value=0, min=0,
                               max=indexed_timeseries.data.shape[0] - 1,
                               orientation='horizontal')
    slider.observe(lambda change: show_image(change.new), names='value')
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
        return show_plane_segmentation(list(img_seg.plane_segmentations.values())[0], neurodata_vis_spec)
    else:
        return neurodata_vis_spec[NWBDataInterface](img_seg, neurodata_vis_spec)


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


def compute_outline(image_mask, threshold):
    x, y = zip(*measure.find_contours(image_mask, threshold)[0])
    return x, y


compute_outline = MemoizeMutable(compute_outline)


def show_plane_segmentation_2d(plane_seg: PlaneSegmentation, color_wheel=color_wheel, color_by=None,
                               threshold=.01, fig=None):
    """

    Parameters
    ----------
    plane_seg: PlaneSegmentation
    color_wheel: list
    color_by: str
    threshold: float
    fig: plotly.graph_objects.Figure, options

    Returns
    -------

    """
    if color_by:
        if color_by in plane_seg:
            cats = np.unique(plane_seg[color_by][:])
        else:
            raise ValueError('specified color_by parameter, {}, not in plane_seg object'.format(color_by))
    data = plane_seg['image_mask'].data
    nUnits = data.shape[0]
    if fig is None:
        fig = go.FigureWidget()
    else:
        fig.data = None
    aux_leg = []

    dummy_trace = go.Scatter(
        x=[None], y=[None],
        name='<b>{}</b>'.format(color_by),
        # set opacity = 0
        line={'color': 'rgba(0, 0, 0, 0)'}
    )
    fig.add_trace(dummy_trace)

    for i in range(nUnits):
        if plane_seg[color_by][i] not in aux_leg:
            show_leg = True
            aux_leg.append(plane_seg[color_by][i])
        else:
            show_leg = False
        kwargs = dict()

        if color_by:
            c = color_wheel[np.where(cats == plane_seg[color_by][i])[0][0]]
            kwargs.update(line_color=c)
        # hover text
        hovertext = '<b>roi_id</b>: ' + str(plane_seg.id[i])
        rois_cols = list(plane_seg.colnames)
        if 'roi_id' in rois_cols:
            rois_cols.remove('roi_id')
        sec_str = '<br>'.join([col + ': ' + str(plane_seg[col][i]) for col in rois_cols if
                               isinstance(plane_seg[col][i], (int, float, np.integer, np.float, str))])
        hovertext += '<br>' + sec_str
        # form cell borders
        x, y = compute_outline(plane_seg['image_mask'][i], threshold)

        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                fill='toself',
                mode='lines',
                name=str(plane_seg[color_by][i]),
                legendgroup=str(plane_seg[color_by][i]),
                showlegend=show_leg,
                text=hovertext,
                hovertext='text',
                line=dict(width=.5),
                **kwargs
            )
        )
        fig.update_layout(
            width=700, height=500,
            margin=go.layout.Margin(l=60, r=60, b=60, t=60, pad=1),
            plot_bgcolor="rgb(245, 245, 245)",
        )
    return fig


def plane_segmentation_2d_widget(plane_seg: PlaneSegmentation, **kwargs):

    categorical_columns = infer_categorical_columns(plane_seg)

    if len(categorical_columns) == 1:
        color_by = list(categorical_columns.keys())[0]
        return show_plane_segmentation_2d(plane_seg, color_by=color_by, **kwargs)

    elif len(categorical_columns) > 1:
        cat_controller = widgets.Dropdown(options=list(categorical_columns), description='color by')

        out_fig = show_plane_segmentation_2d(plane_seg, color_by=cat_controller.value, **kwargs)

        def on_change(change, out_fig=out_fig):
            if change['new'] and isinstance(change['new'], dict):
                ind = change['new']['index']
                if isinstance(ind, int):
                    color_by = change['owner'].options[ind]
                    show_plane_segmentation_2d(plane_seg, color_by=color_by, fig=out_fig, **kwargs)

        cat_controller.observe(on_change)

        return widgets.VBox(children=[cat_controller, out_fig])
    else:
        return show_plane_segmentation_2d(plane_seg, color_by=None, **kwargs)


def show_plane_segmentation(plane_seg: PlaneSegmentation, neurodata_vis_spec: dict):
    if 'voxel_mask' in plane_seg:
        return show_plane_segmentation_3d(plane_seg)
    elif 'image_mask' in plane_seg:
        return plane_segmentation_2d_widget(plane_seg)


def show_grayscale_volume(vol: GrayscaleVolume, neurodata_vis_spec: dict):
    import ipyvolume.pylab as p3

    fig = p3.figure()
    p3.volshow(vol.data, tf=linear_transfer_function([0, 0, 0], max_opacity=.1))
    return fig


class RoiResponseSeriesWidget(BaseGroupedTraceWidget):
    def __init__(self, roi_response_series: RoiResponseSeries, neurodata_vis_spec=None, **kwargs):
        super().__init__(roi_response_series, 'rois', **kwargs)
