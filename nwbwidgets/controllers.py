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


def make_float_range_controller(tmin, tmax, start_value=None):
    if start_value is None:
        start_value = [tmin, min(tmin + 50, tmax)]

    slider = widgets.FloatRangeSlider(
        value=start_value,
        min=tmin,
        max=tmax,
        step=0.1,
        description='time window',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=Layout(width='600px'))

    forward_button = widgets.Button(description='▶', layout=Layout(width='50px'))
    forward_button.on_click(lambda b: move_range_slider_up(slider))

    backwards_button = widgets.Button(description='◀', layout=Layout(width='50px'))
    backwards_button.on_click(lambda b: move_range_slider_down(slider))

    button_box = widgets.HBox(children=[backwards_button, forward_button])
    button_box.layout.align_items = 'center'

    controller = widgets.HBox(children=[slider, button_box])

    return controller


def make_time_window_controller(tmin, tmax, start=0, duration=5.):
    slider = widgets.FloatSlider(
        value=start,
        min=tmin,
        max=tmax,
        step=0.1,
        description='window start (s):',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=Layout(width='100%'))

    duration_widget = widgets.BoundedFloatText(
        value=duration,
        min=0,
        max=tmax - tmin,
        step=0.1,
        description='duration (s):',
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


def make_int_range_controller(max, min=0, start_range=(0, 100), description='units window', orientation='vertical',
                              continuous_update=False):

    slider = widgets.IntRangeSlider(
        value=start_range,
        min=min,
        max=max,
        description=description,
        continuous_update=continuous_update,
        orientation=orientation,
        readout=True,
        style={'description_width': 'initial'},
        layout=Layout(width='100%', height='100%'))

    up_button = widgets.Button(description='▲', layout=Layout(width='50px'))
    up_button.on_click(lambda b: move_range_slider_up(slider))

    down_button = widgets.Button(description='▼', layout=Layout(width='50px'))
    down_button.on_click(lambda b: move_range_slider_down(slider))

    layout = widgets.Layout(display='flex',
                            flex_flow='column',
                            align_items='center')
    controller = widgets.VBox(layout=layout, children=[slider, up_button, down_button])

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
