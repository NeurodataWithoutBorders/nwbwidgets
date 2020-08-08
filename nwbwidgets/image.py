from pathlib import Path, PureWindowsPath

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pynwb
from ipywidgets import widgets, fixed
from pynwb.image import GrayscaleImage, ImageSeries, RGBImage
from tifffile import imread, TiffFile

from .base import fig2widget
from .controllers import StartAndDurationController
from .utils.timeseries import get_timeseries_maxt, get_timeseries_mint


class ImageSeriesWidget(widgets.VBox):
    """Widget showing ImageSeries."""

    def __init__(self, imageseries: ImageSeries,
                 foreign_time_window_controller: StartAndDurationController = None,
                 **kwargs):
        super().__init__()
        self.imageseries = imageseries
        self.controls = {}
        self.out_fig = None

        # Set controller
        if foreign_time_window_controller is None:
            tmin = get_timeseries_mint(imageseries)
            tmax = get_timeseries_maxt(imageseries)
            self.time_window_controller = StartAndDurationController(tmax, tmin)
        else:
            self.time_window_controller = foreign_time_window_controller
        self.set_controls(**kwargs)

        # Make widget figure
        self.set_out_fig()

    def set_controls(self, **kwargs):
        self.controls.update(timeseries=fixed(self.imageseries), time_window=self.time_window_controller)
        self.controls.update({key: widgets.fixed(val) for key, val in kwargs.items()})

    def set_out_fig(self):
        imageseries = self.controls['timeseries'].value
        time_window = self.controls['time_window'].value
        output = widgets.Output()

        if imageseries.external_file is not None:
            file_path = imageseries.external_file[0]
            if "\\" in file_path:
                win_path = PureWindowsPath(file_path)
                path_ext_file = Path(win_path)
            else:
                path_ext_file = Path(file_path)

            # Get Frames dimensions
            tif = TiffFile(path_ext_file)
            n_samples = len(tif.pages)
            page = tif.pages[0]
            n_y, n_x = page.shape

            # Read first frame
            image = imread(path_ext_file, key=0)
            self.out_fig = go.FigureWidget(
                data=go.Heatmap(
                    z=image,
                    colorscale='gray',
                    showscale=False,
                )
            )
            self.out_fig.update_layout(
                xaxis=go.layout.XAxis(showticklabels=False, ticks=""),
                yaxis=go.layout.YAxis(showticklabels=False, ticks=""),
            )

            def on_change(change):
                # Read frame
                mid_timestamp = (change['new'][1] + change['new'][0]) / 2
                frame_number = int(mid_timestamp * imageseries.rate)
                image = imread(path_ext_file, key=frame_number)
                self.out_fig.data[0].z = image

        self.controls['time_window'].observe(on_change)

        self.children = [self.out_fig]


def show_image_series(image_series: ImageSeries, neurodata_vis_spec: dict):
    def show_image(index=0):
        fig, ax = plt.subplots(subplot_kw={'xticks': [], 'yticks': []})
        ax.imshow(image_series.data[index, :, :], cmap='gray')
        fig.show()
        return fig2widget(fig)

    slider = widgets.IntSlider(value=0, min=0,
                               max=image_series.data.shape[0] - 1,
                               orientation='horizontal')
    controls = {'index': slider}
    out_fig = widgets.interactive_output(show_image, controls)
    vbox = widgets.VBox(children=[out_fig, slider])

    return vbox


def show_index_series(index_series, neurodata_vis_spec: dict):
    show_timeseries = neurodata_vis_spec[pynwb.TimeSeries]
    series_widget = show_timeseries(index_series)

    indexed_timeseries = index_series.indexed_timeseries
    image_series_widget = show_image_series(indexed_timeseries,
                                            neurodata_vis_spec)

    return widgets.VBox([series_widget, image_series_widget])


def show_grayscale_image(grayscale_image: GrayscaleImage, neurodata_vis_spec=None):
    fig, ax = plt.subplots()
    plt.imshow(grayscale_image.data[:], 'gray')
    plt.axis('off')

    return fig


def show_rbga_image(rgb_image: RGBImage, neurodata_vis_spec=None):
    fig, ax = plt.subplots()
    plt.imshow(rgb_image.data[:])
    plt.axis('off')

    return fig
