# nwb-jupyter-widgets
Jupyter Widgets for NWB files

authors: Matt McCormick (matt.mccormick@kitware.com) and Ben Dichter (bdichter@lbl.gov)


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

## Demo
![](https://drive.google.com/uc?export=download&id=1JtI2KtT8MielIMvvtgxRzFfBTdc41LiE)


## Features
* Works out-of-the-box on data in NWB:N 2.0
* Can be implemented on a server, so data exploration does not require downloading the data
* Easily extend with custom data types or custom data visualizations
