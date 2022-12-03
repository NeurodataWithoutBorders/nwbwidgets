import ipywidgets as widgets
import plotly.graph_objects as go
import plotly.express as px
from pynwb.ophys import TwoPhotonSeries
from tifffile import imread, TiffFile

from .single_plane import SinglePlaneVisualization
from .plane_slice import PlaneSliceVisualization
from .volume import VolumeVisualization
from ..base import LazyTab


class TwoPhotonSeriesVisualization(widgets.VBox):
    """Widget showing Image stack recorded over time from 2-photon microscope."""

    def __init__(self, indexed_timeseries: TwoPhotonSeries, neurodata_vis_spec: dict):
        super().__init__()

        if indexed_timeseries.data is None:
            if indexed_timeseries.external_file is not None:
                path_ext_file = indexed_timeseries.external_file[0]
                # Get Frames dimensions
                tif = TiffFile(path_ext_file)
                n_samples = len(tif.pages)
                page = tif.pages[0]

                def _add_fig_trace(img_fig: go.Figure, index):
                    if self.figure is None:
                        self.figure = go.FigureWidget(img_fig)
                    else:
                        self.figure.for_each_trace(lambda trace: trace.update(img_fig.data[0]))

                def update_figure(index=0):
                    # Read first frame
                    img_fig = px.imshow(imread(path_ext_file, key=int(index)), binary_string=True)
                    _add_fig_trace(img_fig, index)

                self.slider = widgets.IntSlider(
                    value=0, min=0, max=n_samples - 1, orientation="horizontal", description="TIFF index: "
                )
                self.controls = dict(slider=self.slider)
                self.slider.observe(lambda change: update_figure(index=change.new), names="value")

                update_figure()

                series_name = indexed_timeseries.name
                base_title = f"TwoPhotonSeries: {series_name}"
                self.figure.layout.title = f"{base_title} - read from first external file"

                self.children = [self.figure, self.slider]
        else:
            if len(indexed_timeseries.data.shape) == 3:
                self.children = (SinglePlaneVisualization(two_photon_series=indexed_timeseries),)

            elif len(indexed_timeseries.data.shape) == 4:
                tab = LazyTab(
                    func_dict={"Planar Slice": PlaneSliceVisualization, "3D Volume": VolumeVisualization},
                    data=indexed_timeseries,
                )
                self.children = (tab,)
            else:
                raise NotImplementedError
