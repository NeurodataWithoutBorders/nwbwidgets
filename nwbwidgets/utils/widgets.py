import asyncio

import matplotlib.pyplot as plt

from ipywidgets import Output
from ipywidgets.widgets.interaction import clear_output
import plotly.graph_objects as go


def clean_axes(axes):
    """
    Removes top and right spines from axes

    Parameters
    ----------
    axes: iterable

    """
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def unpack_controls(controls, process_controls=lambda x: x):
    control_states = {}
    for k, v in controls.items():
        # if the value is a dict, add those individually
        if isinstance(v.value, dict):
            control_states.update(v.value)
        else:
            control_states[k] = v.value
    kwargs = process_controls(control_states)
    return kwargs


def interactive_output(f, controls, process_controls=lambda x: x, fixed=None):
    """Connect widget controls to a function.

    This function does not generate a user interface for the widgets (unlike `interact`).
    This enables customisation of the widget user interface layout.
    The user interface layout must be defined and displayed manually.
    """

    if fixed is None:
        fixed = dict()

    out = Output()

    def observer(change):
        with out:
            clear_output(wait=True)
            plot = f(**fixed, **unpack_controls(controls, process_controls))
            plt.show()

    for k, w in controls.items():
        w.observe(observer, "value")
    observer(None)
    return out

def set_plotly_callbacks(f, controls, process_controls=lambda x: x):
    fig = go.FigureWidget()

    def observer(change):
        return f(fig=fig, **unpack_controls(controls, process_controls))
    for k, w in controls.items():
        w.observe(observer, "value")
    return observer(None)

class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback
        self._task = asyncio.ensure_future(self._job())

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def cancel(self):
        self._task.cancel()


def debounce(wait):
    """Decorator that will postpone a function's
    execution until after `wait` seconds
    have elapsed since the last time it was invoked."""

    def decorator(fn):
        timer = None

        def debounced(*args, **kwargs):
            nonlocal timer

            def call_it():
                fn(*args, **kwargs)

            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)

        return debounced

    return decorator
