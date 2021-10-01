import numpy as np
from ipywidgets import widgets, Layout
from tqdm.notebook import tqdm as tqdm_notebook


class ProgressBar(tqdm_notebook):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        # self.container.children[0].layout = Layout(width="80%")


def make_trial_event_controller(trials, layout=None):
    """Controller for which reference to use (e.g. start_time) when making time-aligned averages"""
    trial_events = ["start_time"]
    if not np.all(np.isnan(trials["stop_time"].data)):
        trial_events.append("stop_time")
    trial_events += [
        x.name
        for x in trials.columns
        if (("_time" in x.name) and (x.name not in ("start_time", "stop_time")))
    ]
    kwargs = {}
    if layout is not None:
        kwargs.update(layout=layout)

    trial_event_controller = widgets.SelectMultiple(
        options=trial_events,
        value=["start_time"],
        description='align to:',
        disabled=False,
        **kwargs)
    return trial_event_controller
