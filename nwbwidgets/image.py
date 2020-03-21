import matplotlib.pyplot as plt
import ipywidgets as widgets
import pynwb
from pynwb.image import GrayscaleImage, ImageSeries, RGBImage
from .base import fig2widget


def show_image_series(image_series: ImageSeries, neurodata_vis_spec: dict):

    def show_image(index=0):
        fig, ax = plt.subplots(subplot_kw={'xticks': [], 'yticks': []})
        ax.imshow(image_series.data[index, :, :], cmap='gray')
        plt.show()
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


def show_rbg_image(rgb_image: RGBImage, neurodata_vis_spec=None):
    fig, ax = plt.subplots()
    plt.imshow(rgb_image.data[:])
    plt.axis('off')

    return fig
