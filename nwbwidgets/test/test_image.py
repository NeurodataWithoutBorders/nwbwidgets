import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from nwbwidgets.image import (
    show_rbga_image,
    show_grayscale_image,
    show_index_series,
    show_image_series,
)
from nwbwidgets.view import default_neurodata_vis_spec
from pynwb.base import TimeSeries
from pynwb.image import RGBImage, GrayscaleImage, IndexSeries, ImageSeries


def test_show_rbg_image():
    data = np.random.rand(2700).reshape((30, 30, 3))
    rgb_image = RGBImage(name="test_image", data=data)

    assert isinstance(show_rbga_image(rgb_image), plt.Figure)


def test_show_grayscale_image():
    data = np.random.rand(900).reshape((30, 30))
    grayscale_image = GrayscaleImage(name="test_image", data=data)

    assert isinstance(show_grayscale_image(grayscale_image), plt.Figure)


def test_show_index_series():
    data = np.array([12, 14, 16, 18, 20, 22, 24, 26])
    indexed_timeseries = TimeSeries(
        name="Index Series time data",
        data=np.random.rand(800).reshape((8, 10, 10)),
        rate=1.0,
    )
    index_series = IndexSeries(
        name="Sample Index Series",
        data=data,
        indexed_timeseries=indexed_timeseries,
        rate=1.0,
    )

    assert isinstance(
        show_index_series(index_series, default_neurodata_vis_spec), widgets.Widget
    )


def test_show_image_series():
    data = np.random.rand(800).reshape((8, 10, 10))
    image_series = ImageSeries(name="Image Series", data=data, rate=1.0)

    assert isinstance(
        show_image_series(image_series, default_neurodata_vis_spec), widgets.Widget
    )
