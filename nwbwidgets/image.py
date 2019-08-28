import matplotlib.pyplot as plt
import ipywidgets as widgets
import itkwidgets
import numpy as np
from nwbwidgets import base, view
import pynwb
from pynwb.image import GrayscaleImage
from collections import OrderedDict


def show_image_series(indexed_timeseries, neurodata_vis_spec: OrderedDict):
    output = widgets.Output()

    def show_image(index=0):
        fig, ax = plt.subplots(subplot_kw={'xticks': [], 'yticks': []})
        ax.imshow(indexed_timeseries.data[index][:, :], cmap='gray')
        output.clear_output(wait=True)
        with output:
            plt.show(fig)

    def on_index_change(change):
        show_image(change.new)
    slider = widgets.IntSlider(value=0, min=0,
                               max=indexed_timeseries.data.shape[0] - 1,
                               orientation='horizontal')
    slider.observe(on_index_change, names='value')
    show_image()

    return widgets.VBox([output, slider])


def show_index_series(index_series, neurodata_vis_spec: OrderedDict):
    show_timeseries = neurodata_vis_spec[pynwb.TimeSeries]
    series_widget = show_timeseries(index_series)

    indexed_timeseries = index_series.indexed_timeseries
    image_series_widget = show_image_series(indexed_timeseries,
                                            neurodata_vis_spec)

    return widgets.VBox([series_widget, image_series_widget])


def show_grayscale_image(grayscale_image: GrayscaleImage):
    return itkwidgets.view(np.array(grayscale_image.data),
                           ui_collapsed=True,
                           cmap='Grayscale')
