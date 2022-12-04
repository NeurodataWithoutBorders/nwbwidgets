import math
from functools import lru_cache
from typing import Tuple, Optional

import numpy as np
import h5py

from .single_plane import SinglePlaneVisualization
from .ophys_controllers import VolumetricPlaneSliceController


class PlaneSliceVisualization(SinglePlaneVisualization):
    """Sub-widget specifically for plane-wise views of a 4D TwoPhotonSeries."""

    def _dimension_check(self):
        num_dimensions = len(self.two_photon_series.data.shape)
        if num_dimensions != 4:
            raise ValueError(
                "The PlaneSliceVisualization is only appropriate for "
                f"use on 4-dimensional TwoPhotonSeries! Detected dimension of {num_dimensions}."
            )

    @lru_cache  # default size of 128 items ought to be enough to create a 1GB cache on large images
    def _cache_data_read(self, dataset: h5py.Dataset, frame_index: int, plane_index: int) -> np.ndarray:
        return dataset[frame_index, :, :, plane_index].T

    def update_data(self, frame_index: Optional[int] = None, plane_index: Optional[int] = None):
        frame_index = frame_index or self.Controller.frame_slider.value
        plane_index = plane_index or self.Controller.plane_slider.value

        self.data = self._cache_data_read(
            dataset=self.two_photon_series.data, frame_index=frame_index, plane_index=plane_index
        )

        self.data = self.two_photon_series.data[frame_index, :, :, plane_index]

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
            self.update_data(frame_index=0, plane_index=0)
        else:
            # TOD: Figure out formula for calculating by in one-shot
            by_width = 2
            by_height = 2
            self.data = self.two_photon_series.data[0, ::by_width, ::by_height, 0]

    def pre_setup_controllers(self):
        self.Controller = VolumetricPlaneSliceController()
        self.data_controller_name = "VolumetricDataController"

    def setup_controllers(self):
        """Controller updates are handled through the defined Controller class."""
        super().setup_controllers()

        self.Controller.plane_slider.max = self.two_photon_series.data.shape[-1] - 1

    def update_figure(
        self,
        rotation_changed: Optional[bool] = None,
        frame_index: Optional[int] = None,
        plane_index: Optional[int] = None,
        contrast_rescaling: Optional[str] = None,
        contrast: Optional[Tuple[int]] = None,
    ):
        if plane_index is not None:
            self.update_data(plane_index=plane_index)
            self.update_data_to_plot()

        super().update_figure(rotation_changed=rotation_changed, frame_index=frame_index, contrast_rescaling=contrast_rescaling, contrast=contrast)

    def set_canvas_title(self):
        self.canvas_title = f"TwoPhotonSeries: {self.two_photon_series.name} - Planar slices of volume"

    def setup_observers(self):
        super().setup_observers()

        self.Controller.plane_slider.observe(lambda change: self.update_canvas(plane_index=change.new), names="value")
