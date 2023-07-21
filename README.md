[![PyPI version](https://badge.fury.io/py/nwbwidgets.svg)](https://badge.fury.io/py/nwbwidgets)
[![codecov](https://codecov.io/gh/NeurodataWithoutBorders/nwbwidgets/branch/master/graph/badge.svg)](https://codecov.io/gh/NeurodataWithoutBorders/nwbwidgets)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NeurodataWithoutBorders/nwb-jupyter-widgets/master?filepath=examples%2FNWBWidgets-modality-demos.ipynb)

<p align="center">
  <img src="https://github.com/NeurodataWithoutBorders/nwbwidgets/assets/844306/f20b8c26-79c7-4c1c-a3b5-b49ecf8cce5d" width="350" alt="NWB Widgets"/>
  <h3 align="center">Explore NWB data in Jupyter</h3>
</p>
<p align="center">
   <a href="https://nwb-widgets.readthedocs.io/"><strong>Explore our documentation »</strong></a>
</p>

<!-- TABLE OF CONTENTS -->

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Documentation](#documentation)

## About
A library of widgets for visualization NWB data in a Jupyter notebook (or lab). The widgets allow you to navigate through the hierarchical structure of the NWB file and visualize specific data elements. It is designed to work out-of-the-box with NWB 2.0 files and to be easy to extend.


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
```python
from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget

io = NWBHDF5IO('path/to/file.nwb', mode='r')
nwb = io.read()

nwb2widget(nwb)
```

## Demo
![](https://drive.google.com/uc?export=download&id=1JtI2KtT8MielIMvvtgxRzFfBTdc41LiE)

## Documentation
See our [ReadTheDocs page](https://nwbwidgets.readthedocs.io/en/main/) for full documentation, including a gallery of all supported formats.

## How it works
All visualizations are controlled by the dictionary `neurodata_vis_spec`. The keys of this dictionary are pynwb neurodata types, and the values are functions that take as input that neurodata_type and output a visualization. The visualizations may be of type `Widget` or `matplotlib.Figure`. When you enter a neurodata_type instance into `nwb2widget`, it searches the `neurodata_vis_spec` for that instance's neurodata_type, progressing backwards through the parent classes of the neurodata_type to find the most specific neurodata_type in `neurodata_vis_spec`. Some of these types are containers for other types, and create accordian UI elements for its contents, which are then passed into the `neurodata_vis_spec` and rendered accordingly.

Instead of supplying a function for the value of the `neurodata_vis_spec` dict, you may provide a `dict` or `OrderedDict` with string keys and function values. In this case, a tab structure is rendered, with each of the key/value pairs as an individual tab. All accordian and tab structures are rendered lazily- they are only called with that tab is selected. As a result, you can provide may tabs for a single data type without a worry. They will only be run if they are selected.

## Extending
To extend NWBWidgets, all you need to a function that takes as input an instance of a specific neurodata_type class, and outputs a matplotlib figure or a jupyter widget.

## Used in
* [giocomo-lab-to-nwb](https://github.com/ben-dichter-consulting/giocomo-lab-to-nwb)
* [buffalo-lab-data-to-nwb](https://github.com/ben-dichter-consulting/buffalo-lab-data-to-nwb)
* [axel-lab-to-nwb](https://github.com/ben-dichter-consulting/axel-lab-to-nwb)
