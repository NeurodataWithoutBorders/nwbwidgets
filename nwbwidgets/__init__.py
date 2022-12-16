import plotly.io as pio

from .panel import Panel
from .version import version as __version__
from .view import default_neurodata_vis_spec, nwb2widget

pio.templates.default = "simple_white"
