import plotly.io as pio

from .view import nwb2widget, default_neurodata_vis_spec

# from .ephys_viz_interface import ephys_viz_neurodata_vis_spec

from .version import version as __version__

pio.templates.default = "simple_white"
