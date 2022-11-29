from functools import lru_cache
import numpy as np
from skimage import measure
from multiprocessing import Process, Value
from typing import Tuple, Optional
from functools import lru_cache

import h5py
import ipywidgets as widgets
import plotly.graph_objects as go
import plotly.express as px

from pynwb.base import NWBDataInterface
from pynwb.ophys import (
    RoiResponseSeries,
    DfOverF,
    PlaneSegmentation,
    TwoPhotonSeries,
    ImageSegmentation,
)

from tifffile import imread, TiffFile
from ndx_grayscalevolume import GrayscaleVolume

from .base import df_to_hover_text, LazyTab
from .timeseries import BaseGroupedTraceWidget
from .utils.cmaps import linear_transfer_function
from .utils.dynamictable import infer_categorical_columns
from .controllers import ProgressBar

import plotly.express as px


color_wheel = px.colors.qualitative.Dark24

class TwoPhotonSeriesVolumetricPlaneSliceWidget(widgets.VBox):
    """Sub-widget specifically for plane-wise views of a 4D TwoPhotonSeries."""
    def __init__(self, two_photon_series: TwoPhotonSeries):
        num_dimensions = len(two_photon_series.data.shape)
        if num_dimensions != 4:
            raise ValueError(f"The TwoPhotonSeriesVolumetricPlaneSliceWidget is only appropriate for use on 4-dimensional TwoPhotonSeries! Detected dimension of {num_dimensions}.")

        super().__init__()

        self.two_photon_series = two_photon_series

        self.setup_figure()
        self.setup_controllers()
        self.setup_observers()

        self.children=[self.figure, self.controllers_box]

    def setup_figure(self):
        """Basic setup for the figure layout."""
        self.current_data = self.two_photon_series.data[0, :, :, 0].T

        image = px.imshow(self.current_data, binary_string=True)
        image.update_traces(hovertemplate=None, hoverinfo='skip')
        self.figure = go.FigureWidget(image)
        self.figure.layout.title = f"TwoPhotonSeries: {self.two_photon_series.name} - Planar slices of volume"
        self.figure.update_xaxes(visible=False, showticklabels=False).update_yaxes(visible=False, showticklabels=False)

    def setup_controllers(self):
        """Setup all controllers for the widget."""
        # Frame and plane controllers
        self.frame_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.two_photon_series.data.shape[0] - 1,
            orientation="horizontal",
            description="Frame: ",
            continuous_update=False,
        )
        self.plane_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.two_photon_series.data.shape[-1] - 1,
            orientation="horizontal",
            description="Plane: ",
            continuous_update=False,
        )
        self.frame_and_plane_controller_box = widgets.VBox(children=[self.frame_slider, self.plane_slider])

        # Contrast controllers
        self.manual_contrast_checkbox = widgets.Checkbox(value=False, description="Enable Manual Contrast: ")
        self.auto_contrast_method = widgets.Dropdown(options=["minmax", "infer"], description="Method: ")
        initial_min = np.min(self.current_data)
        initial_max = np.max(self.current_data)
        self.contrast_slider = widgets.IntRangeSlider(
            value=(initial_min, initial_max),
            min=initial_min,
            max=initial_max,
            orientation="horizontal",
            description="Range: ",
            continuous_update=False,
        )
        self.contrast_controller_box = widgets.VBox(children=[self.manual_contrast_checkbox, self.auto_contrast_method])
                          
        self.controllers_box = widgets.HBox(children=[self.frame_and_plane_controller_box, self.contrast_controller_box])

    def setup_observers(self):
        """Given all of the controllers have been initialized and all the update routines have been defined, setup the observer rules for updates on each controller."""
        self.frame_slider.observe(
            lambda change: self.update_plane_slice_figure(frame_index=change.new), names="value"
        )
        self.plane_slider.observe(
            lambda change: self.update_plane_slice_figure(plane_index=change.new), names="value"
        )

        self.manual_contrast_checkbox.observe(
            lambda change: self.switch_contrast_modes(enable_manual_contrast=change.new), names="value"
        )
        self.auto_contrast_method.observe(
            lambda change: self.update_plane_slice_figure(contrast_rescaling=change.new), names="value"
        )
        self.contrast_slider.observe(
            lambda change: self.update_plane_slice_figure(contrast=change.new), names="value"
        )

    def switch_contrast_modes(self, enable_manual_contrast: bool):
        """If the manual contrast checkbox is altered, adjust the manual vs. automatic disabling of the correpsonding controllers."""
        if enable_manual_contrast:
            self.contrast_controller_box.children = [self.manual_contrast_checkbox, self.contrast_slider]
            self.update_plane_slice_figure(contrast=self.contrast_slider.value)
        else:
            self.contrast_controller_box.children = [self.manual_contrast_checkbox, self.auto_contrast_method]
            self.update_plane_slice_figure(contrast_rescaling=self.auto_contrast_method.value)

    def update_contrast_range_per_frame_and_plane(self):
        """
        If either of the frame or plane sliders are changed, be sure to update the valid range of the manual contrast.

        Applies even if current hidden, in case user wants to enable it.
        """
        self.contrast_slider.min = np.min(self.current_data)
        self.contrast_slider.max = np.max(self.current_data)
        self.contrast_slider.value = (max(self.contrast_slider.value[0], self.contrast_slider.min), min(self.contrast_slider.value[1], self.contrast_slider.max))

    @lru_cache # default size of 128 items ought to be enough to create a 1GB cache on large images
    def _cache_data_read(self, dataset: h5py.Dataset, frame_index: int, plane_index: int) -> np.ndarray:
        return dataset[frame_index, :, :, plane_index].T

    def update_data(self, frame_index: int, plane_index: int):
        self.current_data = self._cache_data_read(dataset=self.two_photon_series.data, frame_index=frame_index, plane_index=plane_index)
        self.update_contrast_range_per_frame_and_plane()
        
    def update_plane_slice_figure(
        self,
        frame_index:Optional[int]=None,
        plane_index: Optional[int]=None,
        contrast_rescaling: Optional[str] = None,
        contrast: Optional[Tuple[int]]=None,
    ):
        """Primary update/generation method of the main figure."""
        update_data_region = True if frame_index is not None or plane_index is not None else False

        frame_index = frame_index or self.frame_slider.value
        plane_index = plane_index or self.plane_slider.value
        contrast_rescaling = contrast_rescaling or self.auto_contrast_method.value
        contrast = contrast or self.contrast_slider.value

        if update_data_region:
            self.update_data(frame_index=frame_index, plane_index=plane_index)

        img_fig_kwargs = dict(binary_string=True)
        if self.manual_contrast_checkbox.value:  # Manual contrast
            img_fig_kwargs.update(zmin=contrast[0], zmax=contrast[1])
        else:
            img_fig_kwargs.update(contrast_rescaling=contrast_rescaling)

        image = px.imshow(self.current_data, **img_fig_kwargs)
        image.update_traces(hovertemplate=None, hoverinfo='skip')
        self.figure.data[0].update(image.data[0])

        
class TwoPhotonSeriesWidget(widgets.VBox):
    """Widget showing Image stack recorded over time from 2-photon microscope."""

    def __init__(self, indexed_timeseries: TwoPhotonSeries, neurodata_vis_spec: dict):
        super().__init__()
        self.figure = None
        self.slider = None

        series_name = indexed_timeseries.name
        base_title = f"TwoPhotonSeries: {series_name}"

        def _add_fig_trace(img_fig: go.Figure, index):
            if self.figure is None:
                self.figure = go.FigureWidget(img_fig)
            else:
                self.figure.for_each_trace(lambda trace: trace.update(img_fig.data[0]))

        if indexed_timeseries.data is None:
            if indexed_timeseries.external_file is not None:
                path_ext_file = indexed_timeseries.external_file[0]
                # Get Frames dimensions
                tif = TiffFile(path_ext_file)
                n_samples = len(tif.pages)
                page = tif.pages[0]

                def update_figure(index=0):
                    # Read first frame
                    img_fig = px.imshow(imread(path_ext_file, key=int(index)), binary_string=True)
                    _add_fig_trace(img_fig, index)

                self.slider = widgets.IntSlider(
                    value=0, min=0, max=n_samples - 1, orientation="horizontal", description="TIFF index: "
                )
                self.controls = dict(slider=self.slider)
                self.slider.observe(lambda change: update_figure(index=change.new), names="value")

                update_figure()
                self.figure.layout.title = f"{base_title} - read from first external file"
                self.children = [self.figure, self.slider]
        else:
            self.frame_slider = widgets.IntSlider(
                value=0,
                min=0,
                max=indexed_timeseries.data.shape[0] - 1,
                orientation="horizontal",
                description="Frame: ",
            )
            self.controls = dict(slider=self.frame_slider)

            if len(indexed_timeseries.data.shape) == 3:

                def update_figure(index=0):
                    img_fig = px.imshow(indexed_timeseries.data[index].T, binary_string=True)
                    _add_fig_trace(img_fig, index)

                self.frame_slider.observe(lambda change: update_figure(index=change.new), names="value")

                update_figure()
                self.figure.layout.title = f"{base_title} - planar view"
                self.children = [self.figure, self.frame_slider]

            elif len(indexed_timeseries.data.shape) == 4:

                self.volume_figure = None

                def plot_plane_slices(indexed_timeseries: TwoPhotonSeries):
                    return TwoPhotonSeriesVolumetricPlaneSliceWidget(two_photon_series=indexed_timeseries)

                # Volume tab
                output = widgets.Output()
                def update_volume_figure(index=0):
                    import ipyvolume.pylab as p3

                    p3.figure()
                    p3.volshow(
                        indexed_timeseries.data[index].transpose([1, 0, 2]),
                        tf=linear_transfer_function([0, 0, 0], max_opacity=0.3),
                    )
                    output.clear_output(wait=True)
                    with output:
                        p3.show()

                def first_volume_render(index=0):
                    self.volume_figure = output
                    update_volume_figure(index=self.frame_slider.value)
                    self.frame_slider.observe(lambda change: update_volume_figure(index=change.new), names="value")

                def plot_volume_init(indexed_timeseries: TwoPhotonSeries):
                    self.init_button = widgets.Button(description="Render")
                    self.init_button.on_click(first_volume_render)
                    self.volume_figure = output
                    self.volume_figure.layout.title = f"{base_title} - interactive volume"
                    return widgets.VBox(children=[self.volume_figure, self.frame_slider, self.init_button])

                # Main view
                tab = LazyTab(
                    func_dict={"Planar Slice": plot_plane_slices, "3D Volume": plot_volume_init},
                    data=indexed_timeseries,
                )
                self.children = [tab]
            else:
                raise NotImplementedError


def show_df_over_f(df_over_f: DfOverF, neurodata_vis_spec: dict):
    if len(df_over_f.roi_response_series) == 1:
        title, data_input = list(df_over_f.roi_response_series.items())[0]
        return neurodata_vis_spec[RoiResponseSeries](data_input, neurodata_vis_spec, title=title)
    else:
        return neurodata_vis_spec[NWBDataInterface](df_over_f, neurodata_vis_spec)


def show_image_segmentation(img_seg: ImageSegmentation, neurodata_vis_spec: dict):
    if len(img_seg.plane_segmentations) == 1:
        return route_plane_segmentation(list(img_seg.plane_segmentations.values())[0], neurodata_vis_spec)
    else:
        return neurodata_vis_spec[NWBDataInterface](img_seg, neurodata_vis_spec)


def show_plane_segmentation_3d_voxel(plane_seg: PlaneSegmentation):
    import ipyvolume.pylab as p3

    nrois = len(plane_seg)

    voxel_mask = plane_seg["voxel_mask"]

    mx, my, mz = 0, 0, 0
    for voxel in voxel_mask:
        for x, y, z, _ in voxel:
            mx = max(mx, x)
            my = max(my, y)
            mz = max(mz, z)

    fig = p3.figure()
    for icolor, color in enumerate(color_wheel):
        vol = np.zeros((mx + 1, my + 1, mz + 1))
        sel = np.arange(icolor, nrois, len(color_wheel))
        for isel in sel:
            dat = voxel_mask[isel]
            for x, y, z, value in dat:
                vol[x, y, z] = value
        p3.volshow(vol, tf=linear_transfer_function(color, max_opacity=0.3))
    return fig


def show_plane_segmentation_3d_mask(plane_seg: PlaneSegmentation):
    import ipyvolume.pylab as p3

    nrois = len(plane_seg)

    image_masks = plane_seg["image_mask"]

    fig = p3.figure()
    for icolor, color in enumerate(color_wheel):
        vol = np.zeros(image_masks.shape[1:])
        sel = np.arange(icolor, nrois, len(color_wheel))
        for isel in sel:
            vol += plane_seg["image_mask"][isel]
        p3.volshow(vol, tf=linear_transfer_function(color, max_opacity=0.3))
    return fig


class PlaneSegmentation2DWidget(widgets.VBox):
    def __init__(self, plane_seg: PlaneSegmentation, color_wheel=color_wheel, **kwargs):
        super().__init__()
        self.categorical_columns = infer_categorical_columns(plane_seg)
        self.plane_seg = plane_seg
        self.color_wheel = color_wheel
        self.progress_bar = ProgressBar()
        self.button = widgets.Button(description="Display ROIs")
        self.children = [widgets.HBox([self.button, self.progress_bar.container])]
        self.button.on_click(self.on_button_click)
        self.kwargs = kwargs

    def on_button_click(self, b):
        if len(self.categorical_columns) == 1:
            self.color_by = list(self.categorical_columns.keys())[0]  # changing local variables to instance variables?
            self.children += (self.show_plane_segmentation_2d(color_by=self.color_by, **self.kwargs),)
        elif len(self.categorical_columns) > 1:
            self.cat_controller = widgets.Dropdown(options=list(self.categorical_columns), description="color by")
            self.fig = self.show_plane_segmentation_2d(color_by=self.cat_controller.value, **self.kwargs)

            def on_change(change):
                if change["new"] and isinstance(change["new"], dict):
                    ind = change["new"]["index"]
                    if isinstance(ind, int):
                        color_by = change["owner"].options[ind]
                        self.update_fig(color_by)

            self.cat_controller.observe(on_change)
            self.children += (self.cat_controller, self.fig)
        else:
            self.children += (self.show_plane_segmentation_2d(color_by=None, **self.kwargs),)
        self.children = self.children[1:]

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

    def show_plane_segmentation_2d(
        self,
        color_wheel: list = color_wheel,
        color_by: str = None,
        threshold: float = 0.01,
        fig: go.Figure = None,
        width: int = 600,
        ref_image=None,
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
            if color_by not in self.plane_seg:
                raise ValueError("specified color_by parameter, {}, not in plane_seg object".format(color_by))
            cats = np.unique(self.plane_seg[color_by][:])
            layout_kwargs.update(title=color_by)

        data = self.plane_seg["image_mask"].data
        n_units = len(data)
        if fig is None:
            fig = go.FigureWidget()

        if ref_image is not None:
            fig.add_trace(go.Heatmap(z=ref_image, hoverinfo="skip", showscale=False, colorscale="gray"))

        aux_leg = []
        import pandas as pd

        plane_seg_hover_dict = {
            key: self.plane_seg[key].data for key in self.plane_seg.colnames if key not in ["pixel_mask", "image_mask"]
        }
        plane_seg_hover_dict.update(id=self.plane_seg.id.data)
        plane_seg_hover_df = pd.DataFrame(plane_seg_hover_dict)
        all_hover = df_to_hover_text(plane_seg_hover_df)
        self.progress_bar.reset(total=n_units)
        self.progress_bar.set_description("Loading Image Masks")
        for i in range(n_units):
            kwargs = dict(showlegend=False)
            if color_by is not None:
                if plane_seg_hover_df[color_by][i] not in aux_leg:
                    kwargs.update(showlegend=True)
                    aux_leg.append(plane_seg_hover_df[color_by][i])
                index = np.where(cats == plane_seg_hover_df[color_by][i])[0][0]
                c = color_wheel[index % len(color_wheel)]
                kwargs.update(
                    line_color=c,
                    name=str(plane_seg_hover_df[color_by][i]),
                    legendgroup=str(plane_seg_hover_df[color_by][i]),
                )

            x, y = self.compute_outline(i, threshold)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    fill="toself",
                    mode="lines",
                    text=all_hover[i],
                    hovertext="text",
                    line=dict(width=0.5),
                    **kwargs,
                )
            )
            self.progress_bar.update()
        # self.progress_bar.close()
        fig.update_layout(
            width=width,
            yaxis=dict(
                mirror=True,
                scaleanchor="x",
                scaleratio=1,
                range=[0, self.plane_seg["image_mask"].shape[2]],
                constrain="domain",
            ),
            xaxis=dict(
                mirror=True,
                range=[0, self.plane_seg["image_mask"].shape[1]],
                constrain="domain",
            ),
            margin=dict(t=30, b=10),
            **layout_kwargs,
        )
        return fig

    @lru_cache(1000)
    def compute_outline(self, i, threshold):
        x, y = zip(*measure.find_contours(self.plane_seg["image_mask"][i], threshold)[0])
        return x, y


def route_plane_segmentation(plane_seg: PlaneSegmentation, neurodata_vis_spec: dict):
    if "voxel_mask" in plane_seg:
        return show_plane_segmentation_3d_voxel(plane_seg)
    elif "image_mask" in plane_seg and len(plane_seg.image_mask.shape) == 4:
        raise NotImplementedError("3d image mask vis not implemented yet")
    elif "image_mask" in plane_seg:
        return PlaneSegmentation2DWidget(plane_seg)


def show_grayscale_volume(vol: GrayscaleVolume, neurodata_vis_spec: dict):
    import ipyvolume.pylab as p3

    fig = p3.figure()
    p3.volshow(vol.data, tf=linear_transfer_function([0, 0, 0], max_opacity=0.1))
    return fig


class RoiResponseSeriesWidget(BaseGroupedTraceWidget):
    def __init__(self, roi_response_series: RoiResponseSeries, neurodata_vis_spec=None, **kwargs):
        super().__init__(roi_response_series, "rois", **kwargs)
