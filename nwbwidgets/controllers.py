from ipywidgets import widgets, Layout


def move_slider_up(slider):
    value = slider.get_interact_value()
    value_range = value[1] - value[0]
    max_val = slider.get_state()['max']
    if value[1] + value_range < max_val:
        slider.set_state({'value': (value[0] + value_range, value[1] + value_range)})
    else:
        slider.set_state({'value': (max_val - value_range, max_val)})


def move_slider_down(slider):
    value = slider.get_interact_value()
    value_range = value[1] - value[0]
    min_val = slider.get_state()['min']
    if value[0] - value_range > min_val:
        slider.set_state({'value': (value[0] - value_range, value[1] - value_range)})
    else:
        slider.set_state({'value': (min_val, min_val + value_range)})


def make_time_controller(tmin, tmax, start_value=None):
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
        readout_format='.1f')

    forward_button = widgets.Button(description='▶')
    forward_button.on_click(lambda b: move_slider_up(slider))

    backwards_button = widgets.Button(description='◀')
    backwards_button.on_click(lambda b: move_slider_down(slider))

    controller = widgets.VBox(
        children=[
            slider,
            widgets.HBox(children=[backwards_button, forward_button])])

    return controller


def make_trace_controller(max_val, start_range=(0, 30)):

    slider = widgets.IntRangeSlider(
        value=start_range,
        min=0,
        max=max_val,
        description='units',
        continuous_update=False,
        orientation='horizontal',
        readout=True)

    up_button = widgets.Button(description='▲', layout=Layout(width='auto'))
    up_button.on_click(lambda b: move_slider_up(slider))

    down_button = widgets.Button(description='▼', layout=Layout(width='auto'))
    down_button.on_click(lambda b: move_slider_down(slider))

    controller = widgets.VBox(
        children=[
            slider,
            widgets.VBox(children=[up_button, down_button])])

    return controller
