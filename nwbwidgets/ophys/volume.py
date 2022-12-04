import ipywidgets as widgets
from pynwb.ophys import TwoPhotonSeries

from .ophys_controllers import FrameController
from ..utils.cmaps import linear_transfer_function


class VolumeVisualization(widgets.VBox):
    def _first_volume_render(self):
        self.Canvas = widgets.Output()
        self.update_canvas(index=self.frame_slider.value)

    def __init__(self, two_photon_series: TwoPhotonSeries):
        super().__init__()
        self.two_photon_series = two_photon_series
        self.canvas_title = f"TwoPhotonSeries: {self.two_photon_series.name} - Interactive Volume"

        self.Canvas = widgets.Button(description="Render")
        self.Canvas.on_click(self._first_volume_render)
        self.Canvas.layout.title = self.canvas_title

        self.setup_controllers()
        self.setup_observers()

        self.children = (self.Canvas, self.Controller)

    def update_data(self, frame_index=0):
        self.data = self.two_photon_series.data[frame_index, ...]

    def setup_data(self):
        """Initial data setup could be a lot (> 250 MB), but the true 'setup' of the figure is the 'Render' button."""
        self.update_data(frame_index=0)

    def update_data_to_plot(self):
        self.data_to_plot = self.data.transpose([1, 0, 2])

    def setup_data_to_plot(self):
        self.update_data_to_plot()

    def setup_controllers(self):
        self.Controller = FrameController()

    def update_canvas(self, frame_index: int = 0):
        import ipyvolume.pylab as p3

        p3.figure()
        p3.volshow(self.data_to_plot, tf=linear_transfer_function([0, 0, 0], max_opacity=0.3))
        self.Canvas.clear_output(wait=True)
        with self.Canvas:
            p3.show()

    def setup_canvas(self):
        self.Canvas = widgets.Output()
        self.Canvas.layout.title = self.canvas_title

        self.setup_data()
        self.setup_data_to_plot()
        self.update_canvas()

    def setup_observers(self):
        self.Controller.frame_slider.observe(lambda change: self.update_canvas(frame_index=change.new), names="value")

    def plot_volume_init(self, two_photon_series: TwoPhotonSeries):

        self.volume_figure.layout.title = f"TwoPhotonSeries: {self.two_photon_series.name} - Interactive Volume"
