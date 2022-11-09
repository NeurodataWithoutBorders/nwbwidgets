
# Running NWB widgets on Jupyter

## Using `Panel`

The easiest way to use NWB widgets is with the interactive `Panel`:
```python
from nwbwidgets.panel import Panel

Panel()
```

![panel](../_static/panel.gif)

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

## Using `nwb2widget`

If you're working with a nwb file object in your Jupyter notebook, you can also explore it with NWB Widgets using `nwb2widget`:
```python
from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget

io = NWBHDF5IO('path/to/file.nwb', mode='r')
nwbfile = io.read()

nwb2widget(nwbfile)
```

## Reading remote data
It is also possible to read NWB files directly from S3, e.g. from the DANDI Archive. To identify the http path of 
the s3 url, you will need the dandiset_id, the version and the relative path of the NWB file within that dandiset:

```python
from dandi.dandiapi import DandiAPIClient


dandiset_id = "000006"  # ephys dataset from the Svoboda Lab
version="draft"
filepath = "sub-anm372795/sub-anm372795_ses-20170718.nwb"  # 450 kB file
with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_id, version).get_asset_by_path(filepath)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
```

To read the data, it is recommended to use fsspec and set up a local cache

```python
import fsspec
import pynwb
import h5py
from fsspec.implementations.cached import CachingFileSystem
from nwbwidgets import nwb2widget

# first, create a virtual filesystem based on the http protocol and use
# caching to save accessed data to RAM.
fs = CachingFileSystem(
    fs=fsspec.filesystem("http"),
    cache_storage="nwb-cache",  # Local folder for the cache
)

# next, open the file
f = fs.open(s3_url, "rb")
file = h5py.File(f)
io = pynwb.NWBHDF5IO(file=file, load_namespaces=True)
nwbfile = io.read()

nwb2widget(nwbfile)
```

This approach can be extended to different kinds of remote stores. Learn more about reading data from a remote source
in PyNWB [here](https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/streaming.html).