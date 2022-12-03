import math
from functools import lru_cache
from typing import Tuple, Optional

import h5py
import numpy as np
import plotly.express as px
from pynwb.ophys import TwoPhotonSeries

from .plane_slice import PlaneSliceVisualization


class SinglePlaneVisualization(PlaneSliceVisualization):
    """Sub-widget specifically for plane views of a 3D TwoPhotonSeries."""

    def __init__(self, two_photon_series: TwoPhotonSeries):
        num_dimensions = len(two_photon_series.data.shape)
        if num_dimensions != 3:
            raise ValueError(
                "The SinglePlaneVisualization is only appropriate for "
                f"use on 3-dimensional TwoPhotonSeries! Detected dimension of {num_dimensions}."
            )

        super().__init__()

        # Only major difference with the parent volumetric visualization is the ability to specify the plane index
        # Could remove it altogether, but that's more work than just hiding the component and not using it's value
        self.Controller.plane_slider.layout.visibility = "hidden"
        self.Canvas.layout.title = f"TwoPhotonSeries: {self.two_photon_series.name}"

    @lru_cache  # default size of 128 items ought to be enough to create a 1GB cache on large images
    def _cache_data_read(self, dataset: h5py.Dataset, frame_index: int) -> np.ndarray:
        return dataset[frame_index, :, :].T

    def update_data(self, frame_index: Optional[int] = None):
        frame_index = frame_index or self.Controller.frame_slider.value

        self.data = self._cache_data_read(dataset=self.two_photon_series.data, frame_index=frame_index)

        self.data = self.two_photon_series.data[frame_index, :, :]

        if self.Controller.contrast_type_toggle.value == "Manual":
            self.update_contrast_range()

    def setup_data(self, max_mb_treshold: float = 20.0):
        itemsize = self.two_photon_series.data.dtype.itemsize
        nbytes_per_image = math.prod(self.two_photon_series.data.shape) * itemsize
        if nbytes_per_image <= max_mb_treshold:
            self.update_data(frame_index=0, plane_index=0)
        else:
            # TOD: Figure out formula for calculating by in one-shot
            by_width = 2
            by_height = 2
            self.data = self.two_photon_series.data[0, ::by_width, ::by_height]

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

    def update_canvas(
        self,
        rotation_changed: Optional[bool] = None,
        frame_index: Optional[int] = None,
        contrast_rescaling: Optional[str] = None,
        contrast: Optional[Tuple[int]] = None,
    ):
        self.update_figure(
            rotation_changed=rotation_changed,
            frame_index=frame_index,
            contrast_rescaling=contrast_rescaling,
            contrast=contrast,
        )
        self.Canvas.data[0].update(self.figure.data[0])
