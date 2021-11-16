from pathlib import Path, PureWindowsPath

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pynwb
from ipywidgets import widgets, fixed, Layout
from pynwb.image import GrayscaleImage, ImageSeries, RGBImage
from tifffile import imread, TiffFile

from .base import fig2widget
from .controllers import StartAndDurationController
from .utils.timeseries import (
    get_timeseries_maxt,
    get_timeseries_mint,
    timeseries_time_to_ind,
)


class ImageSeriesWidget(widgets.VBox):
    """Widget showing ImageSeries."""

    def __init__(
        self,
        imageseries: ImageSeries,
        foreign_time_window_controller: StartAndDurationController = None,
        **kwargs
    ):
        super().__init__()
        self.imageseries = imageseries
        self.controls = {}
        self.out_fig = None

        # Set controller
        if foreign_time_window_controller is None:
            tmin = get_timeseries_mint(imageseries)
            if imageseries.external_file and imageseries.rate:
                tif = TiffFile(imageseries.external_file[0])
                tmax = imageseries.starting_time + len(tif.pages) / imageseries.rate
            else:
                tmax = get_timeseries_maxt(imageseries)
            self.time_window_controller = StartAndDurationController(tmax, tmin)
        else:
            self.time_window_controller = foreign_time_window_controller
        self.set_controls(**kwargs)

        # Make widget figure
        self.set_out_fig()

        self.children = [self.out_fig, self.time_window_controller]

    def time_to_index(self, time):
        if self.imageseries.external_file and self.imageseries.rate:
            return int((time - self.imageseries.starting_time) * self.imageseries.rate)
        else:
            return timeseries_time_to_ind(self.imageseries, time)

    def set_controls(self, **kwargs):
        self.controls.update(
            timeseries=fixed(self.imageseries), time_window=self.time_window_controller
        )
        self.controls.update({key: widgets.fixed(val) for key, val in kwargs.items()})

    def get_frame(self, idx):
        if self.imageseries.external_file is not None:
            return imread(self.imageseries.external_file, key=idx)
        else:
            return self.image_series.data[idx].T

    def set_out_fig(self):

        self.out_fig = go.FigureWidget(
            data=go.Heatmap(
                z=self.get_frame(0),
                colorscale="gray",
                showscale=False,
            )
        )
        self.out_fig.update_layout(
            xaxis=go.layout.XAxis(showticklabels=False, ticks=""),
            yaxis=go.layout.YAxis(
                showticklabels=False, ticks="", scaleanchor="x", scaleratio=1
            ),
        )

        def on_change(change):
            # Read frame
            frame_number = self.time_to_index(change["new"][0])
            image = self.get_frame(frame_number)
            self.out_fig.data[0].z = image

        self.controls["time_window"].observe(on_change)


class ImageSeriesWidget(widgets.VBox):

    def __int__(self, image_series: ImageSeries, neurodata_vis_spec: dict = None):
        self.image_series = image_series

        self.index_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=image_series.data.shape[0] - 1,
            orientation="horizontal",
            continuous_update=False,
            description="index",
        )

        if len(image_series.data.shape) == 3:
            self.show_image = self.show_grayscale_image
            self.controls = {"index": self.index_slider}
            out_fig = widgets.interactive_output(self.show_image, self.controls)
            super().__init__(children=(out_fig, self.index_slider))
        else:
            self.show_image = self.show_rgb_image
            self.mode_dropdown = widgets.Dropdown(
                options=("rgb", "bgr"), layout=Layout(width="200px"), description="mode"
            )
            self.controls = {"index": self.index_slider, "mode": self.mode_dropdown}
            out_fig = widgets.interactive_output(self.show_image, self.controls)
            super.__init__(children=(out_fig, self.index_slider, self.mode_dropdown))

    def show_grayscale_image(self, index=0):
        fig, ax = plt.subplots(subplot_kw={"xticks": [], "yticks": []})
        ax.imshow(self.image_series.data[index].T, cmap="gray", aspect="auto")
        return fig2widget(fig)

    def show_rgb_image(self, index=0, mode="rgb"):
        fig, ax = plt.subplots(subplot_kw={"xticks": [], "yticks": []})
        image = self.image_series.data[index]
        if mode == "bgr":
            image = image[:, :, ::-1]
        ax.imshow(image.transpose([1, 0, 2]), aspect="auto")
        return fig2widget(fig)


def show_index_series(index_series, neurodata_vis_spec: dict):
    show_timeseries = neurodata_vis_spec[pynwb.TimeSeries]
    series_widget = show_timeseries(index_series)

    indexed_timeseries = index_series.indexed_timeseries
    image_series_widget = ImageSeriesWidget(indexed_timeseries, neurodata_vis_spec)

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
