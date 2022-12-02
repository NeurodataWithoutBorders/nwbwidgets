import math
from functools import lru_cache
import numpy as np
from skimage import measure
from multiprocessing import Process, Value
from typing import Tuple, Optional, List, Dict
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


color_wheel = px.colors.qualitative.Dark24


class FrameController(widgets.VBox):
    controller_fields = ("frame_slider",)

    def __init__(self):
        super().__init__()

        self.frame_slider = widgets.IntSlider(
            value=0,  # Actual value will depend on data selection
            min=0,  # Actual value will depend on data selection
            max=1,  # Actual value will depend on data selection
            orientation="horizontal",
            description="Frame: ",
            continuous_update=False,
        )
        
        self.children = (self.frame_slider,)

class PlaneController(widgets.VBox):
    controller_fields = ("plane_slider",)

    def __init__(self):
        super().__init__()

        self.plane_slider = widgets.IntSlider(
            value=0,  # Actual value will depend on data selection
            min=0,  # Actual value will depend on data selection
            max=1,  # Actual value will depend on data selection
            orientation="horizontal",
            description="Plane: ",
            continuous_update=False,
        )
        
        self.children = (self.plane_slider,)
        


class ImShowController(widgets.VBox):
    """Controller specifically for handling various options for the plot.express.imshow function."""

    controller_fields = ("manual_contrast_toggle", "auto_contrast_method", "contrast_slider")

    def __init__(self):
        super().__init__()
        
        self.manual_contrast_toggle = widgets.ToggleButtons(
            description="Constrast: ",
            options=[("Automatic", "Automatic"), ("Manual", "Manual")],  # Values set to strings for external readability
        )
        self.auto_contrast_method = widgets.Dropdown(description="Method: ", options=["minmax", "infer"])
        self.contrast_slider = widgets.IntRangeSlider(
            value=(0, 1),  # Actual value will depend on data selection
            min=0,  # Actual value will depend on data selection
            max=1,  # Actual value will depend on data selection
            orientation="horizontal",
            description="Range: ",
            continuous_update=False,
        )

        # Setup initial controller-specific layout
        self.children = (self.manual_contrast_toggle, self.auto_contrast_method)

        # Setup controller-specific observer events
        self.setup_observers()

    def setup_observers(self):
        self.manual_contrast_toggle.observe(
            lambda change: self.switch_contrast_modes(enable_manual_contrast=change.new), names="value"
        )

    def switch_contrast_modes(self, enable_manual_contrast: bool):
        """When the manual contrast toggle is altered, adjust the manual vs. automatic visibility of the components."""
        if self.manual_contrast_toggle.value == "Manual":
            self.children = (self.manual_contrast_toggle, self.contrast_slider)
        elif self.manual_contrast_toggle.value == "Automatic":
            self.children = (self.manual_contrast_toggle, self.auto_contrast_method)

class ViewTypeController(widgets.VBox):
    controller_fields = ("view_type_toggle",)

    def __init__(self):
        super().__init__()

        self.view_type_toggle = widgets.ToggleButtons(
            options=[("Simplified", "Simplified"), ("Detailed", "Detailed")],  # Values set to strings for external readability
        )
        self.children = (self.view_type_toggle,)
            
            
class MultiController(widgets.VBox):
    controller_fields: Tuple[str] = tuple()
    components: Dict[str, widgets.VBox] = dict()
    
    def __init__(self, components: list):
        super().__init__()
        
        children = list()
        controller_fields = list()
        self.components = {component.__class__.__name__: component for component in components}
        for component in self.components.values():
            # Set attributes at outermost level
            for field in component.controller_fields:
                controller_fields.append(field)
                setattr(self, field, getattr(component, field))

            # Default layout of children
            if not isinstance(component, MultiController):
                children.append(component)
                
        self.children = tuple(children)
        self.controller_fields = tuple(controller_fields)
            
        self.setup_observers()

    def setup_observers(self):
        pass

class VolumetricDataController(MultiController):
    def __init__(self):
        super().__init__(components=[FrameController(), PlaneController()])
    
class VolumetricPlaneSliceController(MultiController):
    def __init__(self):
        super().__init__(components=[ViewTypeController(), VolumetricDataController(), ImShowController()])


class PlaneSliceVizualization(widgets.VBox):
    """Sub-widget specifically for plane-wise views of a 4D TwoPhotonSeries."""

    def __init__(self, two_photon_series: TwoPhotonSeries):
        # num_dimensions = len(two_photon_series.data.shape)
        # if num_dimensions != 4:
        #     raise ValueError(
        #         "The TwoPhotonSeriesVolumetricPlaneSliceWidget is only appropriate for "
        #         f"use on 4-dimensional TwoPhotonSeries! Detected dimension of {num_dimensions}."
        #     )

        super().__init__()
        self.two_photon_series = two_photon_series

        self.setup_data()
        self.setup_data_to_plot()

        self.setup_controllers()
        self.setup_canvas()

        self.setup_observers()

        # Setup layout of Canvas relative to Controllers
        self.children = [self.Canvas, self.Controller]

    @lru_cache  # default size of 128 items ought to be enough to create a 1GB cache on large images
    def _cache_data_read(self, dataset: h5py.Dataset, frame_index: int, plane_index: int) -> np.ndarray:
        return dataset[frame_index, :, :, plane_index].T

    def update_contrast_range(self):
        """
        If either of the frame or plane sliders are changed, be sure to update the valid range of the manual contrast.

        Applies even if current hidden, in case user wants to enable it.
        """
        self.Controller.contrast_slider.max = np.max(self.data)
        self.Controller.contrast_slider.min = np.min(self.data)
        self.Controller.contrast_slider.value = (
            max(self.Controller.contrast_slider.value[0], self.Controller.contrast_slider.min),
            min(self.Controller.contrast_slider.value[1], self.Controller.contrast_slider.max),
        )

    def update_data(self, frame_index: Optional[int] = None, plane_index: Optional[int] = None):
        frame_index = frame_index or self.Controller.frame_slider.value
        plane_index = plane_index or self.Controller.plane_slider.value

        self.data = self._cache_data_read(
            dataset=self.two_photon_series.data, frame_index=frame_index, plane_index=plane_index
        )

        self.data = self.two_photon_series.data[frame_index, :, :, plane_index]
        
        if self.Controller.manual_contrast_toggle.value == "Manual":
            self.update_contrast_range()

    def setup_data(self, max_mb_treshold: float = 20.0):
        """
        Start by loading only a single frame of a single plane.

        If the image size relative to data type is too large, relative to max_mb_treshold (indicating the load
        operation for initial setup would take a noticeable amount of time), then sample the image with a `by`.

        Note this may not actually provide a speedup when streaming; need to think of way around that. Maybe set
        a global flag for if streaming mode is enabled on the file, and if so make full use of data within contiguous
        HDF5 chunks?
        """
        itemsize = self.two_photon_series.data.dtype.itemsize
        nbytes_per_image = math.prod(self.two_photon_series.data.shape) * itemsize
        if nbytes_per_image <= max_mb_treshold:
            self.update_data(frame_index=0, plane_index=0)
        else:
            # TOD: Figure out formula for calculating by in one-shot
            by_width = 2
            by_height = 2
            self.data = self.two_photon_series.data[0, ::by_width, ::by_height, 0]

    def update_data_to_plot(self):
        self.data_to_plot = self.data.T

    def setup_data_to_plot(self):
        self.update_data_to_plot()

    def setup_controllers(self):
        """Controller updates are handled through the defined Controller class."""
        self.Controller = VolumetricPlaneSliceController()

        # Setup layout of controllers relative to each other
        self.Controller.children = [
            widgets.VBox(
                children=[
                    self.Controller.components["ViewTypeController"],
                    widgets.HBox(
                        children=[
                            self.Controller.components["VolumetricDataController"],
                            self.Controller.components["ImShowController"],
                        ]
                    ),
                ]
            )
        ]

        # Set some initial values based on neurodata object and initial data to plot
        self.Controller.frame_slider.max = self.two_photon_series.data.shape[0] - 1
        self.Controller.plane_slider.max = self.two_photon_series.data.shape[-1] - 1
        self.Controller.contrast_slider.max = np.max(self.data_to_plot)
        self.Controller.contrast_slider.min = np.min(self.data_to_plot)
        self.Controller.contrast_slider.value = (self.Controller.contrast_slider.min, self.Controller.contrast_slider.max)

    def update_figure(
        self,
        frame_index: Optional[int] = None,
        plane_index: Optional[int] = None,
        contrast_rescaling: Optional[str] = None,
        contrast: Optional[Tuple[int]] = None,
    ):
        if frame_index is not None or plane_index is not None:
            self.update_data(frame_index=frame_index, plane_index=plane_index)
            self.update_data_to_plot()
        
        frame_index = frame_index or self.Controller.frame_slider.value
        plane_index = plane_index or self.Controller.plane_slider.value
        contrast_rescaling = contrast_rescaling or self.Controller.auto_contrast_method.value
        contrast = contrast or self.Controller.contrast_slider.value
        
        img_fig_kwargs = dict(binary_string=True)
        if self.Controller.manual_contrast_toggle.value == "Manual":
            img_fig_kwargs.update(zmin=contrast[0], zmax=contrast[1])
        elif self.Controller.manual_contrast_toggle.value == "Automatic":
            img_fig_kwargs.update(contrast_rescaling=contrast_rescaling)

        self.figure = px.imshow(self.data_to_plot, **img_fig_kwargs)
        self.figure.update_traces(hovertemplate=None, hoverinfo="skip")

    def update_canvas(
        self,
        frame_index: Optional[int] = None,
        plane_index: Optional[int] = None,
        contrast_rescaling: Optional[str] = None,
        contrast: Optional[Tuple[int]] = None,
    ):
        self.update_figure(
            frame_index=frame_index, plane_index=plane_index, contrast_rescaling=contrast_rescaling, contrast=contrast
        )
        self.Canvas.data[0].update(self.figure.data[0])

    def setup_canvas(
        self,
        frame_index: Optional[int] = None,
        plane_index: Optional[int] = None,
        contrast_rescaling: Optional[str] = None,
        contrast: Optional[Tuple[int]] = None,
    ):
        self.update_figure()
        self.Canvas = go.FigureWidget(self.figure)
        self.Canvas.layout.title = f"TwoPhotonSeries: {self.two_photon_series.name} - Planar slices of volume"
        self.Canvas.update_xaxes(visible=False, showticklabels=False).update_yaxes(visible=False, showticklabels=False)

    def setup_observers(self):
        self.Controller.frame_slider.observe(lambda change: self.update_canvas(frame_index=change.new), names="value")
        self.Controller.plane_slider.observe(lambda change: self.update_canvas(plane_index=change.new), names="value")

        self.Controller.view_type_toggle.observe(lambda change: self.update_canvas(), names="value")
        self.Controller.auto_contrast_method.observe(
            lambda change: self.update_canvas(contrast_rescaling=change.new), names="value"
        )
        self.Controller.contrast_slider.observe(lambda change: self.update_canvas(contrast=change.new), names="value")


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
                    return PlaneSliceVizualization(two_photon_series=indexed_timeseries)

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
