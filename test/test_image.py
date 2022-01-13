import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from nwbwidgets.image import (
    show_rbga_image,
    show_grayscale_image,
    show_index_series,
    show_image_series,
    ImageSeriesWidget,
)
from nwbwidgets.view import default_neurodata_vis_spec
from nwbwidgets.controllers.time_window_controllers import StartAndDurationController
from pynwb.base import TimeSeries
from pynwb.image import RGBImage, GrayscaleImage, IndexSeries, ImageSeries
import plotly.graph_objects as go
from .fixtures_imageseries import *


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
        unit="na",
    )
    index_series = IndexSeries(
        name="Sample Index Series",
        data=data,
        indexed_timeseries=indexed_timeseries,
        rate=1.0,
        unit="n.a.",
    )

    assert isinstance(
        show_index_series(index_series, default_neurodata_vis_spec), widgets.Widget
    )


def test_show_image_series():
    data = np.random.rand(800).reshape((8, 10, 10))
    image_series = ImageSeries(name="Image Series", data=data, rate=1.0, unit="n.a.")

    assert isinstance(
        show_image_series(image_series, default_neurodata_vis_spec), widgets.Widget
    )


def test_image_series_widget_data_2d():
    data = np.random.randint(0, 255, size=[10, 30, 40])
    image_series = ImageSeries(name="Image Series", data=data, rate=1.0, unit="n.a.")
    wd = ImageSeriesWidget(image_series)
    assert isinstance(wd.figure, go.FigureWidget)
    assert wd.time_slider.min == 0.0
    assert wd.time_slider.max == 9.0


def test_image_series_widget_data_3d():
    data = np.random.randint(0, 255, size=[10, 30, 40, 5])
    image_series = ImageSeries(name="Image Series", data=data, rate=1.0, unit="n.a.")
    wd = ImageSeriesWidget(image_series)
    assert isinstance(wd.figure, widgets.Output)
    assert wd.time_slider.min == 0.0
    assert wd.time_slider.max == 9.0


def test_image_series_widget_external_file_tif(create_tif_files, movie_no_frames):
    rate = 1.0
    image_series = ImageSeries(
        name="Image Series", external_file=create_tif_files, rate=rate, unit="n.a."
    )
    wd = ImageSeriesWidget(image_series)
    assert isinstance(wd.figure, go.FigureWidget)
    assert wd.time_slider.max == movie_no_frames[0]/rate - 1/rate
    assert wd.time_slider.min == 0.0
    assert wd.file_selector.value == create_tif_files[0]
    wd.file_selector.value = create_tif_files[1]
    assert wd.time_slider.min == movie_no_frames[0]/rate
    assert wd.time_slider.max == sum([movie_no_frames[i]/rate for i in range(len(movie_no_frames))]) - 1/rate


def test_image_series_widget_external_file_single(create_tif_files, movie_no_frames):
    rate = 1.0
    image_series = ImageSeries(
        name="Image Series", external_file=create_tif_files[:1], rate=rate, unit="n.a."
    )
    wd = ImageSeriesWidget(image_series)
    assert isinstance(wd.figure, go.FigureWidget)
    assert wd.time_slider.max == movie_no_frames[0]/rate - 1/rate
    assert wd.time_slider.min == 0.0
    assert wd.file_selector is None


def test_image_series_widget_external_file_video(create_movie_files, movie_no_frames, movie_fps):
    image_series = ImageSeries(
        name="Image Series", external_file=create_movie_files, unit="n.a.", rate=movie_fps
    )
    wd = ImageSeriesWidget(image_series)
    assert isinstance(wd.figure, go.FigureWidget)
    assert wd.time_slider.max == movie_no_frames[0]/movie_fps - 1/movie_fps
    assert wd.time_slider.min == 0.0
    assert wd.file_selector.value == create_movie_files[0]
    wd.file_selector.value = create_movie_files[1]
    assert wd.time_slider.max == sum([movie_no_frames[i]/movie_fps for i in range(len(movie_no_frames))]) - 1/movie_fps
    wd.file_selector.value = create_movie_files[0]
    assert wd.time_slider.max == movie_no_frames[0]/movie_fps - 1/movie_fps
    assert wd.time_slider.min == 0.0


def test_image_series_foreign_time_controller(create_movie_files, movie_no_frames, movie_fps):
    st_controller = StartAndDurationController(tmax=20.0, tmin=0)
    image_series = ImageSeries(
        name="Image Series", external_file=create_movie_files, unit="n.a.", rate=movie_fps
    )
    wd = ImageSeriesWidget(image_series,st_controller)
    assert wd.time_slider.max == movie_no_frames[0]/movie_fps - 1/movie_fps
    st_controller.value = (5.0, 20.0)
    assert wd.time_slider.value == movie_no_frames[0]/movie_fps - 1/movie_fps