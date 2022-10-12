import plotly.io as pio
from .view import nwb2widget, default_neurodata_vis_spec
from .panel import Panel
from .version import version as __version__

pio.templates.default = "simple_white"
