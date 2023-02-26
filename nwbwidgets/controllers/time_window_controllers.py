from abc import abstractmethod
from typing import Optional

from ipywidgets import HBox, Layout, ValueWidget, link, widgets
from ipywidgets.widgets.widget_description import DescriptionWidget


class WindowController(HBox, ValueWidget, DescriptionWidget):
    def __init__(
        self, vmin, vmax, start_value=(None, None), description="window (s)", orientation="horizontal", **kwargs
    ):
        if orientation not in ("horizontal", "vertical"):
            ValueError("Unrecognized orientation: {}".format(orientation))

        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.start_value = start_value
        self.description = description
        self.orientation = orientation

        if self.orientation == "horizontal":
            self.to_start_button = widgets.Button(description="◀◀", layout=Layout(width="65px"))
            self.backwards_button = widgets.Button(description="◀", layout=Layout(width="40px"))
            self.forward_button = widgets.Button(description="▶", layout=Layout(width="40px"))
            self.to_end_button = widgets.Button(description="▶▶", layout=Layout(width="65px"))
        else:  # vertical
            self.to_end_button = widgets.Button(description="▲▲", layout=Layout(width="50px"))
            self.forward_button = widgets.Button(description="▲", layout=Layout(width="50px"))
            self.backwards_button = widgets.Button(description="▼", layout=Layout(width="50px"))
            self.to_start_button = widgets.Button(description="▼▼", layout=Layout(width="50px"))

        self.to_start_button.on_click(self.move_start)
        self.backwards_button.on_click(self.move_down)
        self.forward_button.on_click(self.move_up)
        self.to_end_button.on_click(self.move_end)

    # @abstractmethod
    # @property
    # def value(self):
    #    """Must be window in seconds"""
    #    pass

    @abstractmethod
    def get_children(self):
        pass

    @abstractmethod
    def move_up(self, change):
        pass

    @abstractmethod
    def move_down(self, change):
        pass

    @abstractmethod
    def move_start(self, change):
        pass

    @abstractmethod
    def move_end(self, change):
        pass


class RangeController(WindowController):
    def __init__(
        self,
        vmin,
        vmax,
        start_value=None,
        dtype="float",
        description="time window (s)",
        orientation="horizontal",
        **kwargs,
    ):
        super().__init__(vmin, vmax, start_value, description, orientation, **kwargs)

        self.dtype = dtype
        self.slider = self.make_range_slider(description=description, **kwargs)

        link((self.slider, "value"), (self, "value"))
        link((self.slider, "description"), (self, "description"))

        self.children = self.get_children()

    def make_range_slider(self, **kwargs):
        """

        Parameters
        ----------
        kwargs: passed into RangeSlider constructor

        Returns
        -------

        """

        slider_kwargs = dict(
            value=self.start_value,
            min=self.vmin,
            max=self.vmax,
            continuous_update=False,
            readout=True,
            style={"description_width": "initial"},
            orientation=self.orientation,
        )

        if self.dtype == "float":
            slider_kwargs.update(
                readout_format=".1f",
                step=0.1,
                description="time window (s)",
                layout=Layout(width="100%"),
            )
            slider_kwargs.update(kwargs)
            return widgets.FloatRangeSlider(**slider_kwargs)
        elif self.dtype == "int":
            slider_kwargs.update(description="unit window", layout=Layout(height="100%"))
            slider_kwargs.update(kwargs)
            return widgets.IntRangeSlider(**slider_kwargs)
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

    def get_children(self):
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
    """
    Can be used in place of the RangeController.
    """

    DEFAULT_DURATION = 5

    def __init__(
            self,
            tmax: float,
            tmin: Optional[float] = 0.0,
            start_value: Optional[tuple] = None,
            description: Optional[str] = "start (s)",
            **kwargs,
    ):
        """

        Parameters
        ----------
        tmax: float
            in seconds
        tmin: float, default: 0.0
            in seconds
        start_value: (float, float), optional
            start and stop in seconds
        description: str, default: "start (s)"
        kwargs: dict
        """

        if tmin > tmax:
            raise ValueError("tmax and tmin were probably entered in the wrong order. tmax should be first")

        super().__init__(tmin, tmax, start_value, description, **kwargs)

        self.slider = widgets.FloatSlider(
            value=start_value,
            min=tmin,
            max=tmax,
            step=0.01,
            description=description,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            style={"description_width": "initial"},
            layout=Layout(width="100%", min_width="250px"),
        )

        link((self.slider, "description"), (self, "description"))

        if start_value is None:
            duration = min(self.DEFAULT_DURATION, tmax - tmin)
        else:
            duration = start_value[1] - start_value[0]

        self.duration = widgets.BoundedFloatText(
            value=duration,
            min=0,
            max=tmax - tmin,
            step=0.1,
            description="duration (s):",
            style={"description_width": "initial"},
            layout=Layout(max_width="200px"),
        )

        self.value = (self.slider.value, self.slider.value + self.duration.value)

        self.slider.observe(self.monitor_slider)
        self.duration.observe(self.monitor_duration)

        self.children = self.get_children()

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

    def get_children(self):
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
