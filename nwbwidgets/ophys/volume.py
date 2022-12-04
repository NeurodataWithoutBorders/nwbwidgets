import ipywidgets as widgets
from pynwb.ophys import TwoPhotonSeries

from ..utils.cmaps import linear_transfer_function


class VolumeVisualization(widgets.VBox):
    def __init__(self, two_photon_series: TwoPhotonSeries):
        super().__init__()
        self.two_photon_series = two_photon_series

        self.volume_figure = widgets.Button(description="Render")
        self.children = (self.volume_figure,)

    def update_volume_figure(self, index=0):
        import ipyvolume.pylab as p3

        p3.figure()
        p3.volshow(
            self.two_photon_series.data[index].transpose([1, 0, 2]),
            tf=linear_transfer_function([0, 0, 0], max_opacity=0.3),
        )
        self.volume_figure.clear_output(wait=True)
        with self.volume_figure:
            p3.show()

    def first_volume_render(self, index=0):
        self.volume_figure = widgets.Output()
        self.update_volume_figure(index=self.frame_slider.value)
        self.frame_slider.observe(lambda change: self.update_volume_figure(index=change.new), names="value")

    def plot_volume_init(self, two_photon_series: TwoPhotonSeries):
        self.init_button = widgets.Button(description="Render")
        self.init_button.on_click(self.first_volume_render)

        self.volume_figure.layout.title = f"TwoPhotonSeries: {self.two_photon_series.name} - Interactive Volume"
