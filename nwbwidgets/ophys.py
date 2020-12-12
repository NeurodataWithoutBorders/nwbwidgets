import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from ndx_grayscalevolume import GrayscaleVolume
from pynwb.base import NWBDataInterface
from pynwb.ophys import RoiResponseSeries, DfOverF, PlaneSegmentation, TwoPhotonSeries, ImageSegmentation
from skimage import measure
from tifffile import imread, TiffFile

from .timeseries import BaseGroupedTraceWidget
from .utils.cmaps import linear_transfer_function
from .utils.dynamictable import infer_categorical_columns
from .utils.functional import MemoizeMutable
from .base import df_to_hover_text

color_wheel = ['red', 'blue', 'green', 'black', 'magenta', 'yellow']


class TwoPhotonSeriesWidget(widgets.VBox):
    """Widget showing Image stack recorded over time from 2-photon microscope."""

    def __init__(self, indexed_timeseries: TwoPhotonSeries, neurodata_vis_spec: dict):
        super().__init__()

        output = widgets.Output()

        if indexed_timeseries.data is None:
            if indexed_timeseries.external_file is not None:
                path_ext_file = indexed_timeseries.external_file[0]
                # Get Frames dimensions
                tif = TiffFile(path_ext_file)
                n_samples = len(tif.pages)
                page = tif.pages[0]
                n_y, n_x = page.shape

                def show_image(index=0):
                    fig, ax = plt.subplots(subplot_kw={'xticks': [], 'yticks': []})
                    # Read first frame
                    image = imread(path_ext_file, key=int(index))
                    ax.imshow(image, cmap='gray')
                    output.clear_output(wait=True)
                    with output:
                        fig.show()

                slider = widgets.IntSlider(value=0, min=0,
                                           max=n_samples - 1,
                                           orientation='horizontal')
        else:
            if len(indexed_timeseries.data.shape) == 3:
                def show_image(index=0):
                    fig, ax = plt.subplots(subplot_kw={'xticks': [], 'yticks': []})
                    ax.imshow(indexed_timeseries.data[index], cmap='gray')
                    output.clear_output(wait=True)
                    with output:
                        fig.show()
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
        self.children = [output, slider]


def show_df_over_f(df_over_f: DfOverF, neurodata_vis_spec: dict):
    if len(df_over_f.roi_response_series) == 1:
        title, input = list(df_over_f.roi_response_series.items())[0]
        return neurodata_vis_spec[RoiResponseSeries](input, neurodata_vis_spec, title=title)
    else:
        return neurodata_vis_spec[NWBDataInterface](df_over_f, neurodata_vis_spec)


def show_image_segmentation(img_seg: ImageSegmentation, neurodata_vis_spec: dict):
    if len(img_seg.plane_segmentations) == 1:
        return route_plane_segmentation(list(img_seg.plane_segmentations.values())[0], neurodata_vis_spec)
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


def show_plane_segmentation_2d(
        plane_seg: PlaneSegmentation,
        color_wheel: list = color_wheel,
        color_by: str = None,
        threshold: float = .01,
        fig: go.Figure = None,
        width: int = 600,
        ref_image=None
        ):
    """

    Parameters
    ----------
    plane_seg: PlaneSegmentation
    color_wheel: list, optional
    color_by: str, optional
    threshold: float, optional
    fig: plotly.graph_objects.Figure, optional
    width: int, optional
        width of image in pixels. Height is automatically determined
        to be proportional
    ref_image: image, optional


    Returns
    -------

    """
    layout_kwargs = dict()
    if color_by:
        if color_by not in plane_seg:
            raise ValueError('specified color_by parameter, {}, not in plane_seg object'.format(color_by))
        cats = np.unique(plane_seg[color_by][:])
        layout_kwargs.update(title=color_by)

    data = plane_seg['image_mask'].data
    nUnits = data.shape[0]
    if fig is None:
        fig = go.FigureWidget()

    if ref_image is not None:
        fig.add_trace(
            go.Heatmap(
                z=ref_image,
                hoverinfo='skip',
                showscale=False,
                colorscale='gray'
            )
        )

    aux_leg = []
    all_hover = df_to_hover_text(plane_seg.to_dataframe())

    for i in range(nUnits):
        kwargs = dict(showlegend=False)
        if color_by is not None:
            if plane_seg[color_by][i] not in aux_leg:
                kwargs.update(showlegend=True)
                aux_leg.append(plane_seg[color_by][i])
            c = color_wheel[np.where(cats == plane_seg[color_by][i])[0][0]]
            kwargs.update(line_color=c,
                          name=str(plane_seg[color_by][i]),
                          legendgroup=str(plane_seg[color_by][i]),
                          )

        # form cell borders
        x, y = compute_outline(plane_seg['image_mask'][i], threshold)

        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                fill='toself',
                mode='lines',
                text=all_hover[i],
                hovertext='text',
                line=dict(width=.5),
                **kwargs
            )
        )

    fig.update_layout(
        width=width,
        yaxis=dict(
            mirror=True,
            scaleanchor="x",
            scaleratio=1,
            range=[0, plane_seg['image_mask'].shape[2]],
            constrain='domain'
        ),
        xaxis=dict(
            mirror=True,
            range=[0, plane_seg['image_mask'].shape[1]],
            constrain='domain'
        ),
        margin=dict(t=30, b=10),
        **layout_kwargs
    )
    return fig


class PlaneSegmentation2DWidget(widgets.VBox):
    def __init__(self, plane_seg: PlaneSegmentation, color_wheel=color_wheel, **kwargs):
        super().__init__()
        self.categorical_columns = infer_categorical_columns(plane_seg)
        self.plane_seg = plane_seg
        self.color_wheel = color_wheel

        if len(self.categorical_columns) == 1:
            self.color_by = list(self.categorical_columns.keys())[0]  # changing local variables to instance variables?
            self.children = [show_plane_segmentation_2d(plane_seg, color_by=self.color_by, **kwargs)]
        elif len(self.categorical_columns) > 1:
            self.cat_controller = widgets.Dropdown(options=list(self.categorical_columns), description='color by')
            self.fig = show_plane_segmentation_2d(plane_seg, color_by=self.cat_controller.value, **kwargs)

            def on_change(change):
                if change['new'] and isinstance(change['new'], dict):
                    ind = change['new']['index']
                    if isinstance(ind, int):
                        color_by = change['owner'].options[ind]
                        self.update_fig(color_by)

            self.cat_controller.observe(on_change)
            self.children = [self.cat_controller, self.fig]
        else:
            self.children = [show_plane_segmentation_2d(self.plane_seg, color_by=None, **kwargs)]

    def update_fig(self, color_by):
        cats = np.unique(self.plane_seg[color_by][:])
        legendgroups = []
        with self.fig.batch_update():
            for color_val, data in zip(self.plane_seg[color_by][:], self.fig.data):
                color = self.color_wheel[np.where(cats == color_val)[0][0]]  # store the color
                data.line.color = color  # set the color
                data.legendgroup = str(color_val)  # set the legend group to the color
                data.name = str(color_val)
            for color_val, data in zip(self.plane_seg[color_by][:], self.fig.data):
                if color_val not in legendgroups:
                    data.showlegend = True
                    legendgroups.append(color_val)
                else:
                    data.showlegend = False


def route_plane_segmentation(plane_seg: PlaneSegmentation, neurodata_vis_spec: dict):
    if 'voxel_mask' in plane_seg:
        return show_plane_segmentation_3d(plane_seg)
    elif 'image_mask' in plane_seg:
        return PlaneSegmentation2DWidget(plane_seg)


def show_grayscale_volume(vol: GrayscaleVolume, neurodata_vis_spec: dict):
    import ipyvolume.pylab as p3

    fig = p3.figure()
    p3.volshow(vol.data, tf=linear_transfer_function([0, 0, 0], max_opacity=.1))
    return fig


class RoiResponseSeriesWidget(BaseGroupedTraceWidget):
    def __init__(self, roi_response_series: RoiResponseSeries, neurodata_vis_spec=None, **kwargs):
        super().__init__(roi_response_series, 'rois', **kwargs)
