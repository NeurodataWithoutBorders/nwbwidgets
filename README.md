# nwb-jupyter-widgets
Jupyter Widgets for NWB files. This repo defines a structure for navigating the hierarchical structure with widgets in a jupyter notebook. It is designed to work out-of-the-box with NWB:N 2.0 files and to be easy to extend. Currently most of the visualizations are pretty rudimentary, and we would be happy to work with anyone interested in implementing visualizations.

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
