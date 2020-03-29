from ipywidgets import widgets, Layout
import numpy as np


def move_range_slider_up(slider):
    value = slider.get_interact_value()
    value_range = value[1] - value[0]
    max_val = slider.get_state()['max']
    if value[1] + value_range < max_val:
        slider.set_state({'value': (value[0] + value_range, value[1] + value_range)})
    else:
        slider.set_state({'value': (max_val - value_range, max_val)})


def move_int_slider_up(slider: widgets.IntSlider):
    value = slider.get_interact_value()
    max_val = slider.get_state()['max']
    if value + 1 < max_val:
        slider.value = value + 1


def move_int_slider_down(slider: widgets.IntSlider):
    value = slider.get_interact_value()
    min_val = slider.get_state()['min']
    if value - 1 > min_val:
        slider.value = value - 1


def move_range_slider_down(slider):
    value = slider.get_interact_value()
    value_range = value[1] - value[0]
    min_val = slider.get_state()['min']
    if value[0] - value_range > min_val:
        slider.set_state({'value': (value[0] - value_range, value[1] - value_range)})
    else:
        slider.set_state({'value': (min_val, min_val + value_range)})


def move_slider_up(slider, dur):
    value = slider.get_interact_value()
    max_val = slider.get_state()['max']
    if value + 2 * dur < max_val:
        slider.value = value + dur
    else:
        slider.value = max_val - dur


def move_slider_down(slider,  dur):
    value = slider.get_interact_value()
    min_val = slider.get_state()['min']
    if value - dur > min_val:
        slider.value = value - dur
    else:
        slider.value = min_val


class RangeController(widgets.HBox):

    def __init__(self, vmin, vmax, start_value=None, dtype='float', description='time window (s)',
                 orientation='horizontal', **kwargs):

        if orientation not in ('horizontal', 'vertical'):
            ValueError('Unrecognized orientation: {}'.format(orientation))

        self.vmin = vmin
        self.vmax = vmax
        self.start_value = start_value
        self.orientation = orientation
        self.dtype = dtype

        super(RangeController, self).__init__()

        self.slider = self.make_range_slider(description=description, **kwargs)

        if self.orientation == 'horizontal':
            self.to_start_button = widgets.Button(description='◀◀', layout=Layout(width='55px'))
            self.backwards_button = widgets.Button(description='◀', layout=Layout(width='40px'))
            self.forward_button = widgets.Button(description='▶', layout=Layout(width='40px'))
            self.to_end_button = widgets.Button(description='▶▶', layout=Layout(width='55px'))
        else:  # vertical
            self.to_end_button = widgets.Button(description='▲▲', layout=Layout(width='50px'))
            self.forward_button = widgets.Button(description='▲', layout=Layout(width='40px'))
            self.backwards_button = widgets.Button(description='▼', layout=Layout(width='40px'))
            self.to_start_button = widgets.Button(description='▼▼', layout=Layout(width='50px'))

        self.to_start_button.on_click(self.move_start)
        self.backwards_button.on_click(self.move_down)
        self.forward_button.on_click(self.move_up)
        self.to_end_button.on_click(self.move_end)

        self.children = self.layout()

    def layout(self):
        if self.orientation == 'horizontal':
            return [
                self.slider,
                self.to_start_button,
                self.backwards_button,
                self.forward_button,
                self.to_end_button
            ]
        elif self.orientation == 'vertical':
            return [widgets.VBox([
                self.slider,
                self.to_end_button,
                self.forward_button,
                self.backwards_button,
                self.to_start_button,
            ],
                layout=widgets.Layout(display='flex',
                                      flex_flow='column',
                                      align_items='center')
            )]
        else:
            raise ValueError('Unrecognized orientation: {}'.format(self.orientation))

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
            style={'description_width': 'initial'},
            orientation=self.orientation
        )

        if self.dtype == 'float':
            slider_kwargs.update(
                readout_format='.1f',
                step=0.1,
                description='time window (s)',
                layout=Layout(width='100%')
            )
            slider_kwargs.update(kwargs)
            return widgets.FloatRangeSlider(**slider_kwargs)
        elif self.dtype == 'int':
            slider_kwargs.update(
                description='unit window',
                layout=Layout(height='100%')
            )
            return widgets.IntRangeSlider(**slider_kwargs)
        else:
            raise ValueError('Unrecognized dtype: {}'.format(self.dtype))

    def move_up(self, change):
        value = self.slider.get_interact_value()
        value_range = value[1] - value[0]
        if value[1] + value_range < self.vmax:
            self.slider.set_state({'value': (value[0] + value_range, value[1] + value_range)})
        else:
            self.slider.set_state({'value': (self.vmax - value_range, self.vmax)})

    def move_down(self, change):
        value = self.slider.get_interact_value()
        value_range = value[1] - value[0]
        if value[0] - value_range > self.vmin:
            self.slider.set_state({'value': (value[0] - value_range, value[1] - value_range)})
        else:
            self.slider.set_state({'value': (self.vmin, self.vmin + value_range)})

    def move_start(self, change):
        value = self.slider.get_interact_value()
        value_range = value[1] - value[0]
        self.slider.set_state({'value': (self.vmin, self.vmin + value_range)})

    def move_end(self, change):
        value = self.slider.get_interact_value()
        value_range = value[1] - value[0]
        self.slider.set_state({'value': (self.vmax - value_range, self.vmax)})


def make_time_window_controller(tmin, tmax, start=0, duration=5.):
    slider = widgets.FloatSlider(
        value=start,
        min=tmin,
        max=tmax,
        step=0.1,
        description='window start (s)',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        style={'description_width': 'initial'},
        layout=Layout(width='100%'))

    duration_widget = widgets.BoundedFloatText(
        value=duration,
        min=0,
        max=tmax - tmin,
        step=0.1,
        description='duration (s):',
        style={'description_width': 'initial'},
        layout=Layout(width='150px')
    )

    forward_button = widgets.Button(description='▶', layout=Layout(width='50px'))
    forward_button.on_click(lambda b: move_slider_up(slider, duration_widget.get_interact_value()))

    backwards_button = widgets.Button(description='◀', layout=Layout(width='50px'))
    backwards_button.on_click(lambda b: move_slider_down(slider, duration_widget.get_interact_value()))

    controller = widgets.HBox(
        children=[slider,
                  duration_widget,
                  backwards_button,
                  forward_button])

    return controller


def int_controller(max, min=0, value=0, description='unit', orientation='horizontal', continuous_update=False):
    slider = widgets.IntSlider(
        value=value,
        min=min,
        max=max,
        description=description,
        continuous_update=continuous_update,
        orientation=orientation,
        readout=True
    )

    up_button = widgets.Button(description='▲', layout=Layout(width='40px', height='20px'))
    up_button.on_click(lambda b: move_int_slider_up(slider))

    down_button = widgets.Button(description='▼', layout=Layout(width='40px', height='20px'))
    down_button.on_click(lambda b: move_int_slider_down(slider))

    controller = widgets.HBox(
        children=[
            slider,
            widgets.VBox(children=[up_button, down_button])])

    return controller


def make_trial_event_controller(trials):
    trial_events = ['start_time']
    if not np.all(np.isnan(trials['stop_time'].data)):
        trial_events.append('stop_time')
    trial_events += [x.name for x in trials.columns if
                     (('_time' in x.name) and (x.name not in ('start_time', 'stop_time')))]
    trial_event_controller = widgets.Dropdown(options=trial_events,
                                              value='start_time',
                                              description='align to: ')
    return trial_event_controller
