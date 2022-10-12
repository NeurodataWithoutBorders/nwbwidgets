
# Quickstart

The easiest way to use NWB widgets is:
```python
from nwbwidgets.panel import panel

panel()
```

[GIF]

## Basic usage with local files

In your Jupyter notebook, run:
```python
from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget

io = NWBHDF5IO('path/to/file.nwb', mode='r')
nwb = io.read()

nwb2widget(nwb)
```