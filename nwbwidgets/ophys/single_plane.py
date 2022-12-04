import math
from functools import lru_cache
from typing import Tuple, Optional

import h5py
import numpy as np
import ipywidgets as widgets
import plotly.express as px
import plotly.graph_objects as go
from pynwb.ophys import TwoPhotonSeries

from .ophys_controllers import SinglePlaneSliceController


class SinglePlaneVisualization(widgets.VBox):
    """Sub-widget specifically for plane views of a 3D TwoPhotonSeries."""

    def _dimension_check(self):
        num_dimensions = len(self.two_photon_series.data.shape)
        if num_dimensions != 3:
            raise ValueError(
                "The SinglePlaneVisualization2 is only appropriate for "
                f"use on 3-dimensional TwoPhotonSeries! Detected dimension of {num_dimensions}."
            )

    def __init__(self, two_photon_series: TwoPhotonSeries):
        self.two_photon_series = two_photon_series
        self._dimension_check()

        super().__init__()

        self.setup_data()
        self.setup_data_to_plot()

        self.setup_controllers()
        self.canvas_title = f"TwoPhotonSeries: {self.two_photon_series.name} - Planar slices of volume"
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
        self.Controller.manual_contrast_slider.max = np.max(self.data)
        self.Controller.manual_contrast_slider.min = np.min(self.data)
        self.Controller.manual_contrast_slider.value = (
            max(self.Controller.manual_contrast_slider.value[0], self.Controller.manual_contrast_slider.min),
            min(self.Controller.manual_contrast_slider.value[1], self.Controller.manual_contrast_slider.max),
        )

    def update_data(self, frame_index: Optional[int] = None):
        frame_index = frame_index or self.Controller.frame_slider.value

        self.data = self._cache_data_read(dataset=self.two_photon_series.data, frame_index=frame_index)

        self.data = self.two_photon_series.data[frame_index, :, :]

        if self.Controller.contrast_type_toggle.value == "Manual":
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
            self.update_data(frame_index=0)
        else:
            # TOD: Figure out formula for calculating by in one-shot
            by_width = 2
            by_height = 2
            self.data = self.two_photon_series.data[0, ::by_width, ::by_height]

    def get_rotation(self) -> int:
        """The rotation attribute of the SinglePlaneDataController cannot be attached in a modifiable state."""
        if not hasattr(self, "Controller"):  # First time this is called
            return 0
        return self.Controller.components[self.data_controller_name].components["RotationController"].rotation

    def update_data_to_plot(self):
        rotation = self.get_rotation()
        rotation_mod = rotation % 4  # Only supporting 90 degree increments
        if rotation_mod == 0:
            self.data_to_plot = self.data
        elif rotation_mod == 1:
            self.data_to_plot = self.data.T
        elif rotation_mod == 2:
            self.data_to_plot = np.flip(self.data)
        elif rotation_mod == 3:
            self.data_to_plot = np.flip(self.data.T)

    def setup_data_to_plot(self):
        self.update_data_to_plot()

    def pre_setup_controllers(self):
        """This can change in child classes."""
        self.Controller = SinglePlaneSliceController()
        self.data_controller_name = "SinglePlaneDataController"

    def setup_controllers(self):
        """Controller updates are handled through the defined Controller class."""
        self.pre_setup_controllers()

        # Setup layout of controllers relative to each other
        self.Controller.children = [
            widgets.VBox(
                children=[
                    self.Controller.components["ViewTypeController"],
                    widgets.HBox(
                        children=[
                            self.Controller.components[self.data_controller_name],  # Can change in child classes
                            self.Controller.components["ImShowController"],
                        ]
                    ),
                ]
            )
        ]

        # Set some initial values based on neurodata object and initial data to plot
        self.Controller.frame_slider.max = self.two_photon_series.data.shape[0] - 1
        self.Controller.manual_contrast_slider.max = np.max(self.data_to_plot)
        self.Controller.manual_contrast_slider.min = np.min(self.data_to_plot)
        self.Controller.manual_contrast_slider.value = (
            self.Controller.manual_contrast_slider.min,
            self.Controller.manual_contrast_slider.max,
        )

    def update_figure(
        self,
        rotation_changed: Optional[bool] = None,
        frame_index: Optional[int] = None,
        contrast_rescaling: Optional[str] = None,
        contrast: Optional[Tuple[int]] = None,
    ):
        if rotation_changed is not None:
            self.update_data_to_plot()
        elif frame_index is not None:
            self.update_data(frame_index=frame_index)
            self.update_data_to_plot()

        contrast_rescaling = contrast_rescaling or self.Controller.auto_contrast_method.value
        contrast = contrast or self.Controller.manual_contrast_slider.value

        img_fig_kwargs = dict(binary_string=True)
        if self.Controller.contrast_type_toggle.value == "Manual":
            img_fig_kwargs.update(zmin=contrast[0], zmax=contrast[1])
        elif self.Controller.contrast_type_toggle.value == "Automatic":
            img_fig_kwargs.update(contrast_rescaling=contrast_rescaling)

        self.figure = px.imshow(self.data_to_plot, **img_fig_kwargs)
        self.figure.update_traces(hovertemplate=None, hoverinfo="skip")

    def update_canvas(self, **update_figure_kwargs):
        self.update_figure(**update_figure_kwargs)
        self.Canvas.data[0].update(self.figure.data[0])

    def set_canvas_title(self):
        """This can change in child classes."""
        self.canvas_title = f"TwoPhotonSeries: {self.two_photon_series.name}"

    def setup_canvas(self):
        # Setup main figure area
        self.update_figure()
        self.Canvas = go.FigureWidget(self.figure)
        self.set_canvas_title()
        self.Canvas.layout.title = self.canvas_title
        self.Canvas.update_xaxes(visible=False, showticklabels=False).update_yaxes(visible=False, showticklabels=False)

        # Final vizualization-specific setup of controller positions
        # Move the Simplified/Detailed switch to the right part of screen
        self.Controller.components["ViewTypeController"].layout.align_items = "flex-end"

    def setup_observers(self):
        self.Controller.rotate_right.on_click(lambda change: self.update_canvas(rotation_changed=True))
        self.Controller.rotate_left.on_click(lambda change: self.update_canvas(rotation_changed=True))
        self.Controller.frame_slider.observe(lambda change: self.update_canvas(frame_index=change.new), names="value")

        self.Controller.contrast_type_toggle.observe(lambda change: self.update_canvas(), names="value")
        self.Controller.auto_contrast_method.observe(
            lambda change: self.update_canvas(contrast_rescaling=change.new), names="value"
        )
        self.Controller.manual_contrast_slider.observe(
            lambda change: self.update_canvas(contrast=change.new), names="value"
        )
