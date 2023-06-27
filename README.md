# NWB Widgets
A library of widgets for visualization NWB data in a Jupyter notebook (or lab). The widgets allow you to navigate through the hierarchical structure of the NWB file and visualize specific data elements. It is designed to work out-of-the-box with NWB 2.0 files and to be easy to extend.

[![PyPI version](https://badge.fury.io/py/nwbwidgets.svg)](https://badge.fury.io/py/nwbwidgets)
[![codecov](https://codecov.io/gh/NeurodataWithoutBorders/nwbwidgets/branch/master/graph/badge.svg)](https://codecov.io/gh/NeurodataWithoutBorders/nwbwidgets)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NeurodataWithoutBorders/nwb-jupyter-widgets/master?filepath=examples%2FNWBWidgets-modality-demos.ipynb)



## Installation

`nwbwidgets` requires Python >= 3.7.

The latest published version can be installed by running:

```bash
pip install nwbwidgets
```

Note that there are some optional dependencies required for some widgets.
If an NWB data file contains a data type that requires additional dependencies,
you will see a list of extra modules needed for that specific widget.
All other widgets in the file will still work.

## Usage

### Using `Panel`
The easiest way to use NWB widgets is with the interactive `Panel`:

```python
from nwbwidgets.panel import Panel

Panel()
```

### Using `nwb2widget`
If you’re working directly with a NWB file object in your Jupyter notebook, you can also explore it with NWB Widgets using 

```python
from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget

io = NWBHDF5IO('path/to/file.nwb', mode='r')
nwb = io.read()

nwb2widget(nwb)
```

### Using Docker
You can also run the NWB Widgets Panel using Docker:

```bash
$ docker run -p 8866:8866 ghcr.io/NeurodataWithoutBorders/nwbwidgets-voila:latest
```

## Demo
![](https://drive.google.com/uc?export=download&id=1JtI2KtT8MielIMvvtgxRzFfBTdc41LiE)

## How it works
All visualizations are controlled by the dictionary `neurodata_vis_spec`. The keys of this dictionary are pynwb neurodata types, and the values are functions that take as input that neurodata_type and output a visualization. The visualizations may be of type `Widget` or `matplotlib.Figure`. When you enter a neurodata_type instance into `nwb2widget`, it searches the `neurodata_vis_spec` for that instance's neurodata_type, progressing backwards through the parent classes of the neurodata_type to find the most specific neurodata_type in `neurodata_vis_spec`. Some of these types are containers for other types, and create accordian UI elements for its contents, which are then passed into the `neurodata_vis_spec` and rendered accordingly.

Instead of supplying a function for the value of the `neurodata_vis_spec` dict, you may provide a `dict` or `OrderedDict` with string keys and function values. In this case, a tab structure is rendered, with each of the key/value pairs as an individual tab. All accordian and tab structures are rendered lazily- they are only called with that tab is selected. As a result, you can provide may tabs for a single data type without a worry. They will only be run if they are selected.

## Extending
To extend NWBWidgets, all you need to a function that takes as input an instance of a specific neurodata_type class, and outputs a matplotlib figure or a jupyter widget.

## Used in
* [giocomo-lab-to-nwb](https://github.com/ben-dichter-consulting/giocomo-lab-to-nwb)
* [buffalo-lab-data-to-nwb](https://github.com/ben-dichter-consulting/buffalo-lab-data-to-nwb)
* [axel-lab-to-nwb](https://github.com/ben-dichter-consulting/axel-lab-to-nwb)
