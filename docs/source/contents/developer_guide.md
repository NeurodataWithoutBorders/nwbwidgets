
# Developer's guide

All visualizations are controlled by the dictionary `neurodata_vis_spec`. The keys of this dictionary are pynwb neurodata types, and the values are functions that take as input that neurodata_type and output a visualization object. The visualizations may be of type `ipywidgets.widgets.widget.Widget`, `matplotlib.Figure` or `plotly.graph_objects.Figure`. When you enter a neurodata_type instance into nwb2widget, it searches the neurodata_vis_spec for that instance's neurodata_type, progressing backwards through the parent classes of the neurodata_type to find the most specific neurodata_type in neurodata_vis_spec. Some of these types are containers for other types, and create accordion UI elements for its contents, which are then passed into the `neurodata_vis_spec` and rendered accordingly.

Instead of supplying a function for the value of the `neurodata_vis_spec` dict, you may provide a `dict` or `OrderedDict` with string keys and function values. In this case, a tab structure is rendered, with each of the key/value pairs as an individual tab. All accordion and tab structures are rendered lazily- they are only called with that tab is selected. As a result, you can provide may tabs for a single data type without a worry. They will only be run if they are selected.


## Extending and creating new Widgets
<!-- TODO - use an example for new widgets creation-->
To extend NWBWidgets, all you need to a function that takes as input an instance of a specific neurodata_type class, and outputs a matplotlib figure or a jupyter widget
