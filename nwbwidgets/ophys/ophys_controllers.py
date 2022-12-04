import ipywidgets as widgets

from ..controllers import RotationController, ImShowController, ViewTypeController, MultiController


class FrameController(widgets.VBox):
    controller_fields = ("frame_slider",)

    def __init__(self):
        super().__init__()

        self.frame_slider = widgets.IntSlider(
            value=0,  # Actual value will depend on data selection
            min=0,  # Actual value will depend on data selection
            max=1,  # Actual value will depend on data selection
            orientation="horizontal",
            description="Frame: ",
            continuous_update=False,
        )

        self.children = (self.frame_slider,)


class PlaneController(widgets.VBox):
    controller_fields = ("plane_slider",)

    def __init__(self):
        super().__init__()

        self.plane_slider = widgets.IntSlider(
            value=0,  # Actual value will depend on data selection
            min=0,  # Actual value will depend on data selection
            max=1,  # Actual value will depend on data selection
            orientation="horizontal",
            description="Plane: ",
            continuous_update=False,
        )

        self.children = (self.plane_slider,)


class VolumetricDataController(MultiController):
    def __init__(self):
        super().__init__(components=[RotationController(), FrameController(), PlaneController()])

        # Align rotation buttons to center of sliders
        self.layout.align_items = "center"


class VolumetricPlaneSliceController(MultiController):
    def __init__(self):
        super().__init__(components=[ViewTypeController(), VolumetricDataController(), ImShowController()])

        self.setup_visibility()
        self.setup_observers()

    def set_detailed_visibility(self, visibile: bool):
        widget_visibility_type = "visible" if visibile else "hidden"

        self.contrast_type_toggle.layout.visibility = widget_visibility_type
        self.manual_contrast_slider.layout.visibility = widget_visibility_type
        self.auto_contrast_method.layout.visibility = widget_visibility_type

    def update_visibility(self):
        if self.view_type_toggle.value == "Simplified":
            self.set_detailed_visibility(visibile=False)
        elif self.view_type_toggle.value == "Detailed":
            self.set_detailed_visibility(visibile=True)

    def setup_visibility(self):
        self.set_detailed_visibility(visibile=False)

    def setup_observers(self):
        self.view_type_toggle.observe(lambda change: self.update_visibility())
