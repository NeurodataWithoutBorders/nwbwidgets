
# Quickstart

The easiest way to use NWB widgets is with the interactive `Panel`:
```python
from nwbwidgets.panel import Panel

Panel()
```

![panel](../_static/quickstart/example_ecephys.gif)

## Basic usage with local files

In your Jupyter notebook, run:
```python
from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget

io = NWBHDF5IO('path/to/file.nwb', mode='r')
nwb = io.read()

nwb2widget(nwb)
```