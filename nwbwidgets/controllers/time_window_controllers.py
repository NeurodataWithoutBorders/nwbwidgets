from typing import Tuple, Optional

from ipywidgets import Layout, ValueWidget, link, widgets
from ipywidgets.widgets.widget_description import DescriptionWidget

from .basecontroller import BaseController


class WindowController(BaseController, ValueWidget, DescriptionWidget):
    def __init__(
        self,
        vmin: int,
        vmax: int,
        start_value: Optional[Tuple[Optional[int], Optional[int]]] = (None, None),
        description: str = "window (s)",
        orientation: str = "horizontal",  # TODO: should be Literal["horizontal", "vertical"]
        **kwargs,
    ):
        self.vmin = vmin
        self.vmax = vmax
        self.start_value = start_value
        self.description = description
        self.orientation = orientation

        super().__init__()

    def setup_attributes(self):
        if self.orientation == "horizontal":
            self.to_start_button = widgets.Button(description="◀◀", layout=Layout(width="65px"))
            self.backwards_button = widgets.Button(description="◀", layout=Layout(width="40px"))
            self.forward_button = widgets.Button(description="▶", layout=Layout(width="40px"))
            self.to_end_button = widgets.Button(description="▶▶", layout=Layout(width="65px"))
        elif self.orientation == "vertical":
            self.to_end_button = widgets.Button(description="▲▲", layout=Layout(width="50px"))
            self.forward_button = widgets.Button(description="▲", layout=Layout(width="50px"))
            self.backwards_button = widgets.Button(description="▼", layout=Layout(width="50px"))
            self.to_start_button = widgets.Button(description="▼▼", layout=Layout(width="50px"))
        else:
            raise ValueError(
                f"Unrecognized orientation '{self.orientation}', should be either 'horizontal' or 'vertical'"
            )

    def move_up(self, change):
        raise NotImplementedError("The 'move_up' method of this WindowController has not been defined.")

    def move_down(self, change):
        raise NotImplementedError("The 'move_down' method of this WindowController has not been defined.")

    def move_start(self, change):
        raise NotImplementedError("The 'move_start' method of this WindowController has not been defined.")

    def move_end(self, change):
        raise NotImplementedError("The 'move_end' method of this WindowController has not been defined.")

    def setup_observers(self):
        self.to_start_button.on_click(self.move_start)
        self.backwards_button.on_click(self.move_down)
        self.forward_button.on_click(self.move_up)
        self.to_end_button.on_click(self.move_end)

    def get_children(self) -> widgets.Box:
        raise NotImplementedError("The 'get_children' method of this WindowController has not been defined.")

    def setup_children(self):
        self.children = self.get_children()


class RangeController(WindowController):
    def __init__(
        self,
        vmin: int,
        vmax: int,
        start_value: Optional[Tuple[Optional[int], Optional[int]]] = None,
        dtype: str = "float",
        description: str = "time window (s)",
        orientation: str = "horizontal",  # TODO: should be Literal["horizontal", "vertical"]
        **slider_kwargs,
    ):
        self.dtype = dtype

        super().__init__(
            vmin=vmin,
            vmax=vmax,
            start_value=start_value,
            description=description,
            orientation=orientation,
            **slider_kwargs,
        )

        link((self.slider, "value"), (self, "value"))
        link((self.slider, "description"), (self, "description"))

    def setup_attributes(
        self,
        **slider_kwargs,
    ):
        super().setup_attributes()

        default_slider_kwargs = dict(
            value=self.start_value,
            min=self.vmin,
            max=self.vmax,
            continuous_update=False,
            readout=True,
            style={"description_width": "initial"},
            orientation=self.orientation,
        )

        if self.dtype == "float":
            default_slider_kwargs.update(
                readout_format=".1f",
                step=0.1,
                description="time window (s)",
                layout=Layout(width="100%"),
            )
            default_slider_kwargs.update(slider_kwargs)
            self.slider = widgets.FloatRangeSlider(**default_slider_kwargs)
        elif self.dtype == "int":
            default_slider_kwargs.update(description="unit window", layout=Layout(height="100%"))
            default_slider_kwargs.update(slider_kwargs)
            self.slider = widgets.IntRangeSlider(**default_slider_kwargs)
        else:
            raise ValueError("Unrecognized dtype: {}".format(self.dtype))

    def move_up(self, change):
        value_range = self.value[1] - self.value[0]
        if self.value[1] + value_range < self.vmax:
            self.value = (self.value[0] + value_range, self.value[1] + value_range)
        else:
            self.move_end(change)

    def move_down(self, change):
        value_range = self.value[1] - self.value[0]
        if self.value[0] - value_range > self.vmin:
            self.value = (self.value[0] - value_range, self.value[1] - value_range)
        else:
            self.move_start(change)

    def move_start(self, change):
        value_range = self.value[1] - self.value[0]
        self.value = (self.vmin, self.vmin + value_range)

    def move_end(self, change):
        value_range = self.value[1] - self.value[0]
        self.value = (self.vmax - value_range, self.vmax)

    def get_children(self) -> widgets.Box:

        if self.orientation == "horizontal":
            return [
                self.slider,
                self.to_start_button,
                self.forward_button,
                self.backwards_button,
                self.to_end_button,
            ]
        else:
            return [
                widgets.VBox(
                    [
                        self.slider,
                        self.to_end_button,
                        self.forward_button,
                        self.backwards_button,
                        self.to_start_button,
                    ],
                    layout=widgets.Layout(display="flex", flex_flow="column", align_items="center"),
                )
            ]


class StartAndDurationController(WindowController):
    """Can be used in place of the RangeController."""

    DEFAULT_DURATION = 5

    def __init__(
        self,
        tmax: int,
        tmin: int = 0,
        start_value: Optional[Tuple[Optional[int], Optional[int]]] = None,
        description: str = "start (s)",
        orientation: str = "horizontal",  # TODO: should be Literal["horizontal", "vertical"]
    ):
        """
        Parameters
        ----------
        tmax: float
            in seconds
        tmin: float
            in seconds
        start_value: (float, float)
            start and stop in seconds
        description: str
        orientation: str
        """
        if tmin > tmax:
            raise ValueError("tmax and tmin were probably entered in the wrong order. tmax should be first")

        super().__init__(
            vmin=tmin, vmax=tmax, start_value=start_value, description=description, orientation=orientation
        )

        link((self.slider, "description"), (self, "description"))

    def setup_attributes(self):
        super().setup_attributes()

        if self.start_value is None:
            duration = min(self.DEFAULT_DURATION, self.vmax - self.vmin)
        else:
            duration = self.start_value[1] - self.start_value[0]

        self.slider = widgets.FloatSlider(
            value=self.start_value,
            min=self.vmin,
            max=self.vmax,
            step=0.01,
            description=self.description,
            continuous_update=False,
            orientation=self.orientation,
            readout=True,
            readout_format=".2f",
            style={"description_width": "initial"},
            layout=Layout(width="100%", min_width="250px"),
        )

        self.duration = widgets.BoundedFloatText(
            value=duration,
            min=0,
            max=self.vmax - self.vmin,
            step=0.1,
            description="duration (s):",
            style={"description_width": "initial"},
            layout=Layout(max_width="200px"),
        )

        self.value = (self.slider.value, self.slider.value + self.duration.value)

    def move_up(self, change):
        if self.slider.value + 2 * self.duration.value < self.vmax:
            self.slider.value += self.duration.value
        else:
            self.move_end(change)

    def move_down(self, change):
        if self.slider.value - self.duration.value > self.vmin:
            self.slider.value -= self.duration.value
        else:
            self.move_start(change)

    def move_start(self, change):
        self.slider.value = self.vmin

    def move_end(self, change):
        self.slider.value = self.vmax - self.duration.value

    def monitor_slider(self, change):
        if "new" in change:
            if isinstance(change["new"], dict):
                if "value" in change["new"]:
                    value = change["new"]["value"]
                else:
                    return
            else:
                value = change["new"]
        if self.slider.value + self.duration.value > self.vmax:
            self.slider.value = self.vmax - self.duration.value
        else:
            self.value = (value, value + self.duration.value)

    def monitor_duration(self, change):
        if "new" in change:
            if isinstance(change["new"], dict):
                if "value" in change["new"]:
                    value = change["new"]["value"]
                    if self.slider.value + value > self.vmax:
                        self.slider.value = self.vmax - value
                    self.value = (self.slider.value, self.slider.value + value)

    def setup_observers(self):
        super().setup_observers()

        self.slider.observe(self.monitor_slider)
        self.duration.observe(self.monitor_duration)

    def get_children(self) -> widgets.Box:

        if self.orientation == "horizontal":
            return [
                self.slider,
                self.duration,
                # self.to_start_button,
                self.backwards_button,
                self.forward_button,
                # self.to_end_button
            ]
        else:
            return [
                widgets.VBox(
                    [
                        self.slider,
                        self.duration,
                        # self.to_end_button,
                        self.forward_button,
                        self.backwards_button,
                        # self.to_start_button,
                    ],
                    layout=widgets.Layout(display="flex", flex_flow="column", align_items="center"),
                )
            ]
