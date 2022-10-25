
# Quickstart

## Basic usage with `Panel`

The easiest way to use NWB widgets is with the interactive `Panel`:
```python
from nwbwidgets.panel import Panel

Panel()
```

![panel](../_static/quickstart/panel.gif)

With `Panel` you can easily browser through local files as well as stream remote datasets from DANDI archive.
For data streaming the default mode is [fsspec](https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/streaming.html#streaming-method-2-fsspec). If you would like to use [ROS3](https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/streaming.html#streaming-method-1-ros3) instead, you can do so with: 
```python
Panel(stream_mode='ros3')
```

If you intend to you `Panel` only for local storage (no streaming), you can instantiate it as:
```python
Panel(enable_dandi_source=False, enable_s3_source=False)
```

If you intend to you `Panel` only for streaming data (no local storage), you can instantiate it as:
```python
Panel(enable_local_source=False)
```

## Basic usage with `nwb2widget`

If you're working with a nwb file object in your Jupyter notebook, you can also explore it with NWB Widgets using `nwb2widget`:
```python
from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget

io = NWBHDF5IO('path/to/file.nwb', mode='r')
nwbfile = io.read()

nwb2widget(nwbfile)
```

This option will also work if the `nwbfile` object is streaming data from a [remote source](https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/streaming.html).