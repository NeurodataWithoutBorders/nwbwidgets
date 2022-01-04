from pathlib import Path, PureWindowsPath

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pynwb
from ipywidgets import widgets, fixed, Layout
from pynwb.image import GrayscaleImage, ImageSeries, RGBImage

from .base import fig2widget
from .controllers import StartAndDurationController
from .utils.timeseries import (
    get_timeseries_maxt,
    get_timeseries_mint,
    timeseries_time_to_ind,
)
from .utils.cmaps import linear_transfer_function
from typing import Union

try:
    import cv2
    HAVE_OPENCV = True
except ImportError:
    HAVE_OPENCV = False

try:
    from tifffile import imread, TiffFile
    HAVE_TIF = True
except ImportError:
    HAVE_TIF = False

PathType = Union[str, Path]


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

        tmin = get_timeseries_mint(imageseries)

        # Make widget figure --------
        def _add_fig_trace(img_fig: go.Figure, index):
            if self.figure is None:
                self.figure = go.FigureWidget(img_fig)
            else:
                self.figure.for_each_trace(lambda trace: trace.update(img_fig.data[0]))
            self.figure.layout.title = f"Frame no: {index}"

        if imageseries.external_file is not None:
            file_selector = widgets.Dropdown(options=imageseries.external_file)
            path_ext_file = file_selector.value
            tmax = imageseries.starting_time + get_frame_count(path_ext_file)/imageseries.rate
            # Get Frames dimensions
            def update_figure(index=0):
                # Read first frame
                img_fig = px.imshow(get_frame(path_ext_file, index), binary_string=True)
                _add_fig_trace(img_fig, index)

            if foreign_time_window_controller is None:
                self.time_window_controller = StartAndDurationController(tmax, tmin)
            else:
                self.time_window_controller = foreign_time_window_controller
            self.set_children(file_selector)
        else:
            if len(imageseries.data.shape) == 3:

                def update_figure(index=0):
                    img_fig = px.imshow(
                        imageseries.data[index].T, binary_string=True
                    )
                    _add_fig_trace(img_fig, index)

            elif len(imageseries.data.shape) == 4:
                import ipyvolume.pylab as p3

                output = widgets.Output()

                def update_figure(index=0):
                    p3.figure()
                    p3.volshow(
                        imageseries.data[index].transpose([1, 0, 2]),
                        tf=linear_transfer_function([0, 0, 0], max_opacity=0.3),
                    )
                    output.clear_output(wait=True)
                    self.figure = output
                    with output:
                        p3.show()

            else:
                raise NotImplementedError
            tmax = get_timeseries_maxt(imageseries)
            if foreign_time_window_controller is None:
                self.time_window_controller = StartAndDurationController(tmax, tmin)
            else:
                self.time_window_controller = foreign_time_window_controller
            self.set_children()

        self.set_controls(**kwargs)
        self.figure = None

        def on_change(change):
            # Read frame
            frame_number = self.time_to_index(change["new"][0])
            update_figure(frame_number)

        update_figure()
        self.controls["time_window"].observe(on_change)

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

    def set_children(self, *args):
        self.children = [self.out_fig,
                         self.time_window_controller,
                         *args]

    def get_frame(self, idx):
        if self.imageseries.external_file is not None:
            return get_frame(self.imageseries.external_file[0])
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

def get_frame_shape(external_path_file: PathType):
    external_path_file = Path(external_path_file)
    if external_path_file.suffix in ['tif','tiff']:
        assert HAVE_TIF, 'pip install tifffile'
        tif = TiffFile(external_path_file)
        page = tif.pages[0]
        return page.shape
    else:
        assert HAVE_OPENCV, 'pip install opencv-python'
        cap = cv2.VideoCapture(str(external_path_file))
        success, frame = cap.read()
        cap.release()
        return frame.shape

def get_frame_count(external_path_file: PathType):
    external_path_file = Path(external_path_file)
    if external_path_file.suffix in ['tif', 'tiff']:
        assert HAVE_TIF, 'pip install tifffile'
        tif = TiffFile(external_path_file)
        return len(tif.pages)
    else:
        assert HAVE_OPENCV, 'pip install opencv-python'
        cap = cv2.VideoCapture(str(external_path_file))
        if int(cv2.__version__.split(".")[0]) < 3:
            frame_count_arg = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        else:
            frame_count_arg = cv2.CAP_PROP_FRAME_COUNT
        frame_count = cap.get(frame_count_arg)
        cap.release()
        return frame_count

def get_frame(external_path_file: PathType, index):
    external_path_file = Path(external_path_file)
    if external_path_file.suffix in ['tif','tiff']:
        assert HAVE_TIF, 'pip install tifffile'
        return imread(str(external_path_file), key=int(index))
    else:
        assert HAVE_OPENCV, 'pip install opencv-python'
        cap = cv2.VideoCapture(str(external_path_file))
        if int(cv2.__version__.split(".")[0]) < 3:
            set_arg = cv2.cv.CV_CAP_PROP_POS_FRAMES
        else:
            set_arg = cv2.CAP_PROP_POS_FRAMES
        set_value = cap.set(set_arg, index)
        success, frame = cap.read()
        cap.release()
        return frame