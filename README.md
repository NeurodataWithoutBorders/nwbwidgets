# nwb-jupyter-widgets
Jupyter Widgets for NWB files


## Installation
```bash
pip install nwb-jupyter-widgets
```

## Usage
```python
from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget

io = NWBHDF5IO('path/to/file.nwb', mode='r')
nwb = io.read()

nwb2widget(nwb)
```

![](https://drive.google.com/uc?export=download&id=1JtI2KtT8MielIMvvtgxRzFfBTdc41LiE)