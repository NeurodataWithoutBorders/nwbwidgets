from pathlib import Path
from typing import Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pynwb
from ipywidgets import widgets, Layout
from pynwb.image import GrayscaleImage, ImageSeries, RGBImage

from .base import fig2widget
from .utils.cmaps import linear_transfer_function
from .utils.imageseries import get_frame_count, get_frame, get_fps
from .utils.timeseries import (
    get_timeseries_maxt,
    get_timeseries_mint,
    timeseries_time_to_ind,
)
from .controllers.time_window_controllers import StartAndDurationController
PathType = Union[str, Path]


class ImageSeriesWidget(widgets.VBox):
    """Widget showing ImageSeries."""

    def __init__(
        self,
        imageseries: ImageSeries,
        foreign_start_duration_controller: StartAndDurationController = None,
        neurodata_vis_spec: dict = None,
    ):
        super().__init__()
        self.imageseries = imageseries
        self.figure = None
        self.time_slider = None
        self.external_file = None
        self.file_selector = None
        self.video_start_times = []

        if imageseries.external_file is not None:
            self.video_start_times = self._get_video_start_times()
            self.time_slider = widgets.FloatSlider(
                min=self.video_start_times[0],
                max=self.video_start_times[1],
                orientation="horizontal",
                description="time(s)",
                continuous_update=False,
            )
            self.external_file = imageseries.external_file[0]
            # set file selector:
            if len(imageseries.external_file) > 1:
                self.file_selector = widgets.Dropdown(options=imageseries.external_file)
                self.external_file = self.file_selector.value
                self.file_selector.observe(self._update_time_slider, names="value")

            self.time_slider.observe(self._time_slider_callback_external, names="value")
            self._set_figure_from_time(
                imageseries.starting_time, imageseries.starting_time, self.external_file,
            )
        else:
            tmin = get_timeseries_mint(imageseries)
            tmax = get_timeseries_maxt(imageseries)
            self.time_slider = widgets.FloatSlider(
                value=tmin,
                min=tmin,
                max=tmax,
                orientation="horizontal",
                description="time(s)",
                continuous_update=False,
            )
            if len(imageseries.data.shape) == 3:
                self._set_figure_2d(0)
                self.time_slider.observe(self._time_slider_callback_2d, names="value")

            elif len(imageseries.data.shape) == 4:
                self._set_figure_3d(0)
                self.time_slider.observe(self._time_slider_callback_3d, names="value")
            else:
                raise NotImplementedError

        # set visible time slider:
        if foreign_start_duration_controller is None:
            self.visible_time_slider = self.time_slider
        else:
            self.visible_time_slider = foreign_start_duration_controller
            # link the value[0] to time_slider value
            def _link_time_slider(change):
                self.time_slider.value = change["new"][0]
            self.visible_time_slider.observe(_link_time_slider, names="value")
        self.children = self.get_children(self.file_selector)

    def _get_video_start_times(self):
        if self.external_file is not None:
            start_times=[self.imageseries.starting_time]
            for file in tqdm(self.imageseries.external_file,
                             desc="retrieving video start times"):
                fps = get_fps(file)
                fps = fps if fps is not None else self.imageseries.rate
                file_time_duration = get_frame_count(file) / fps
                start_times.append(file_time_duration)
            return np.cumsum(start_times)

    def _time_slider_callback_2d(self, change):
        self._set_figure_from_time(change["new"][0])

    def _time_slider_callback_3d(self, change):
        frame_number = self.time_to_index(change["new"][0])
        self._set_figure_3d(frame_number)

    def _time_slider_callback_external(self, change):
        time = change["new"]
        starting_time = change["owner"].min
        self._set_figure_from_time(time, starting_time, self.external_file)

    def _update_time_slider(self, value):
        path_ext_file = value["new"]
        self.external_file = path_ext_file
        idx = self.imageseries.external_file.index(self.external_file)
        tmin = self.video_start_times[idx]
        tmax = self.video_start_times[idx+1]
        self.time_slider.min = tmin
        self.time_slider.max = tmax
        self._set_figure_from_frame(0, self.external_file)

    def _set_figure_3d(self, frame_number):
        import ipyvolume.pylab as p3

        output = widgets.Output()
        p3.figure()
        p3.volshow(
            self.imageseries.data[frame_number].transpose([1, 0, 2]),
            tf=linear_transfer_function([0, 0, 0], max_opacity=0.3),
        )
        output.clear_output(wait=True)
        self.figure = output
        with output:
            p3.show()

    def _set_figure_from_time(self, time, starting_time, ext_file_path=None):
        frame_number = self.time_to_index(time, starting_time)
        self._set_figure_from_frame(frame_number, ext_file_path)

    def _set_figure_from_frame(self, frame_number, ext_file_path=None):
        data = self.get_frame(frame_number, ext_file_path)
        if self.figure is None:
            img = px.imshow(data, binary_string=True)
            self.figure = go.FigureWidget(img)
        else:
            img = px.imshow(data, binary_string=True)
            self.figure.for_each_trace(lambda trace: trace.update(img.data[0]))
            self.figure.layout.title = f"Frame no: {frame_number}"

    def time_to_index(self, time, starting_time=None):
        starting_time = (
            starting_time
            if starting_time is not None
            else self.imageseries.starting_time
        )
        if self.imageseries.external_file and self.imageseries.rate:
            return int((time - starting_time) * self.imageseries.rate)
        else:
            return timeseries_time_to_ind(self.imageseries, time)

    def get_children(self, *widgets):
        set_widgets = [wid for wid in widgets if wid is not None]
        return [self.figure, self.visible_time_slider, *set_widgets]

    def get_frame(self, idx, ext_file_path=None):
        if ext_file_path is not None:
            return get_frame(ext_file_path)
        else:
            return self.imageseries.data[idx].T


def show_image_series(image_series: ImageSeries, neurodata_vis_spec: dict):
    if len(image_series.data.shape) == 3:
        return show_grayscale_image_series(image_series, neurodata_vis_spec)

    def show_image(index=0, mode="rgb"):
        fig, ax = plt.subplots(subplot_kw={"xticks": [], "yticks": []})
        image = image_series.data[index]
        if mode == "bgr":
            image = image[:, :, ::-1]
        ax.imshow(image.transpose([1, 0, 2]), cmap="gray", aspect="auto")
        fig.show()
        return fig2widget(fig)

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=image_series.data.shape[0] - 1,
        orientation="horizontal",
        continuous_update=False,
        description="index",
    )
    mode = widgets.Dropdown(
        options=("rgb", "bgr"), layout=Layout(width="200px"), description="mode"
    )
    controls = {"index": slider, "mode": mode}
    out_fig = widgets.interactive_output(show_image, controls)
    vbox = widgets.VBox(children=[out_fig, slider, mode])

    return vbox


def show_grayscale_image_series(image_series: ImageSeries, neurodata_vis_spec: dict):
    def show_image(index=0):
        fig, ax = plt.subplots(subplot_kw={"xticks": [], "yticks": []})
        ax.imshow(image_series.data[index].T, cmap="gray", aspect="auto")
        return fig

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=image_series.data.shape[0] - 1,
        orientation="horizontal",
        continuous_update=False,
        description="index",
    )
    controls = {"index": slider}
    out_fig = widgets.interactive_output(show_image, controls)
    vbox = widgets.VBox(children=[out_fig, slider])

    return vbox


def show_index_series(index_series, neurodata_vis_spec: dict):
    show_timeseries = neurodata_vis_spec[pynwb.TimeSeries]
    series_widget = show_timeseries(index_series)

    indexed_timeseries = index_series.indexed_timeseries
    image_series_widget = show_image_series(indexed_timeseries, neurodata_vis_spec)

    return widgets.VBox([series_widget, image_series_widget])


def show_grayscale_image(grayscale_image: GrayscaleImage, neurodata_vis_spec=None):
    fig, ax = plt.subplots()
    plt.imshow(grayscale_image.data[:].T, "gray")
    plt.axis("off")

    return fig


def show_rbga_image(rgb_image: RGBImage, neurodata_vis_spec=None):
    fig, ax = plt.subplots()
    plt.imshow(rgb_image.data[:].transpose([1, 0, 2]))
    plt.axis("off")

    return fig
