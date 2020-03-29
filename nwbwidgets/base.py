from nwbwidgets import view
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython import display
from collections.abc import Iterable
from pynwb import ProcessingModule
from pynwb.core import NWBDataInterface
from matplotlib.pyplot import Figure
from datetime import datetime
from typing import Union
import pandas as pd

GroupingWidget = Union[widgets.Accordion, widgets.Tab]


def show_fields(node, **kwargs) -> widgets.Widget:
    field_lay = widgets.Layout(max_height='40px', max_width='600px',
                               min_height='30px', min_width='130px')
    info = []
    for key, val in node.fields.items():
        lbl_key = widgets.Label(key+':', layout=field_lay)
        lbl_val = widgets.Label(str(val), layout=field_lay)
        info.append(widgets.HBox(children=[lbl_key, lbl_val]))
    vbox = widgets.VBox(info)
    return vbox


# def show_dynamic_table(node: DynamicTable, **kwargs):
def show_dynamic_table(node, **kwargs) -> widgets.Widget:
    out1 = widgets.Output()
    with out1:
        display.display(node.to_dataframe())
    return out1


def show_neurodata_base(node: NWBDataInterface, neurodata_vis_spec: dict) -> widgets.Widget:
    """
    Gets a pynwb object and returns a Vertical Box containing textual info and
    an expandable Accordion with it's children.
    """
    field_lay = widgets.Layout(max_height='40px', max_width='500px',
                               min_height='30px', min_width='180px')
    info = []         # string data type, exposed as a Text widget
    neuro_data = []   # more complex data types, also with children
    labels = []
    for key, value in node.fields.items():
        if isinstance(value, (str, datetime)):
            lbl_key = widgets.Label(key+':', layout=field_lay)
            lbl_val = widgets.Label(str(value), layout=field_lay)
            info.append(widgets.HBox(children=[lbl_key, lbl_val]))
        elif key == 'related_publications':
            pub_list = []
            for pub in value:
                pub_list.append(widgets.HTML(value="<a href=http://dx.doi.org/"+pub[4:]+">"+pub+"</a>"))
            lbl_key = widgets.Label(key+':', layout=field_lay)
            pub_list.insert(0, lbl_key)
            info.append(widgets.HBox(children=pub_list))
        elif key == 'experimenter':
            lbl_experimenter = widgets.Label('Experimenter:', layout=field_lay)
            if isinstance(value, (list, tuple)):
                lbl_names = widgets.Label(', '.join(value), layout=field_lay)
            else:
                lbl_names = widgets.Label(value, layout=field_lay)
            hbox_exp = widgets.HBox(children=[lbl_experimenter, lbl_names])
            info.append(hbox_exp)
        elif (isinstance(value, Iterable) and len(value)) or value:
            neuro_data.append(view.nwb2widget(value, neurodata_vis_spec=neurodata_vis_spec))
            labels.append(key)
    accordion = widgets.Accordion(children=neuro_data, selected_index=None)
    for i, label in enumerate(labels):
        if hasattr(node.fields[label], 'description') and node.fields[label].description:
            accordion.set_title(i, label + ': ' + node.fields[label].description)
        else:
            accordion.set_title(i, label)
    return widgets.VBox(info + [accordion])


def dict2accordion(d: dict, neurodata_vis_spec: dict, **pass_kwargs) -> widgets.Accordion:
    children = [widgets.HTML('Rendering...') for _ in d]
    accordion = widgets.Accordion(children=children, selected_index=None)
    for i, label in enumerate(d):
        if hasattr(d[label], 'description') and d[label].description:
            accordion.set_title(i, label + ': ' + d[label].description)
        else:
            accordion.set_title(i, label)
        accordion.set_title(i, label)

    def on_selected_index(change):
        if change.new is not None and isinstance(change.owner.children[change.new], widgets.HTML):
            children[change.new] = nwb2widget(list(d.values())[change.new], neurodata_vis_spec=neurodata_vis_spec,
                                              **pass_kwargs)
            change.owner.children = children

    accordion.observe(on_selected_index, names='selected_index')

    return accordion


def lazy_tabs(in_dict: dict, node, style: GroupingWidget = widgets.Tab) -> GroupingWidget:
    """Creates a lazy tab object where multiple visualizations can be used for a single node and are generated on the
    fly

    Parameters
    ----------
    in_dict: dict
        keys are labels for tabs and values are functions
    node: NWBDataInterface
        instance of neurodata type to visualize
    style: ipywidgets.Tab or ipywidgets.Accordion, optional
        which way to present the data

    Returns
    -------
    tab: widget

    """
    tabs_spec = list(in_dict.items())

    children = [tabs_spec[0][1](node)] + [widgets.HTML('Rendering...')
                                          for _ in range(len(tabs_spec) - 1)]
    tab = style(children=children)
    [tab.set_title(i, label) for i, (label, _) in enumerate(tabs_spec)]

    def on_selected_index(change):
        if isinstance(change.owner.children[change.new], widgets.HTML):
            children[change.new] = vis2widget(tabs_spec[change.new][1](node))
            change.owner.children = children

    tab.observe(on_selected_index, names='selected_index')

    return tab


def lazy_show_over_data(list_, func_, labels=None, style: GroupingWidget = widgets.Tab) -> GroupingWidget:
    """
    Apply same function to list of data in lazy tabs or lazy accordion
    Parameters
    ----------
    list_
    func_
    labels: list of str
    style: widgets.Tab or widgets.Accordion

    Returns
    -------
    ipywidgets.Tab or ipywidgets.Accordion
        subtype Tab or Accordion

    """
    children = [vis2widget(func_(list_[0]))] + [widgets.HTML('Rendering...') for _ in range(len(list_) - 1)]
    out = style(children=children)
    if labels is not None:
        [out.set_title(i, label) for i, label in enumerate(labels)]

    def on_selected_index(change):
        if change.new is not None and isinstance(change.owner.children[change.new], widgets.HTML):
            children[change.new] = vis2widget(func_(list_[change.new]))
            change.owner.children = children

    out.observe(on_selected_index, names='selected_index')

    return out


def nwb2widget(node,  neurodata_vis_spec: dict, **pass_kwargs) -> widgets.Widget:
    for ndtype in type(node).__mro__:
        if ndtype in neurodata_vis_spec:
            spec = neurodata_vis_spec[ndtype]
            if isinstance(spec, dict):
                return lazy_tabs(spec, node)
            elif callable(spec):
                return vis2widget(spec(node, neurodata_vis_spec=neurodata_vis_spec, **pass_kwargs))
    out1 = widgets.Output()
    with out1:
        print(node)
    return out1


def vis2widget(vis) -> widgets.Widget:
    if isinstance(vis, widgets.Widget):
        return vis
    elif isinstance(vis, plt.Figure):
        return fig2widget(vis)
    elif isinstance(vis, plt.Axes):
        return fig2widget(vis.get_figure())
    else:
        raise ValueError('unsupported vis type {}'.format(type(vis)))


def fig2widget(fig: Figure, **kwargs) -> widgets.Widget:
    out = widgets.Output()
    with out:
        plt.show(fig)
    return out


def processing_module(node: ProcessingModule, neurodata_vis_spec: dict) -> widgets.Widget:
    return nwb2widget(node.data_interfaces, neurodata_vis_spec=neurodata_vis_spec)


def show_text_fields(node, exclude=('comments', 'interval'), **kwargs) -> widgets.Widget:
    info = []
    for key in node.fields:
        if key not in exclude and isinstance(key, (str, float, int)):
            info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    return widgets.VBox(info)


def df2accordion(df: pd.DataFrame, by, func, style: GroupingWidget = widgets.Accordion, detect_single=True) \
        -> GroupingWidget:
    """
    Visualize pandas.DataFrame with an ipywidgets.Accordion

    Parameters
    ----------
    df: pandas.DataFrame
    by: str
    func: visualization function
    style: ipywidgets.Tab or ipywidgets.Accordion, optional
    detect_single: bool
        If True, test if the dimension you are grouping by only has 1 unique value. If so, do not form an Accordion.

    Returns
    -------
    ipywigets.Accordion or ipywidgets.Tab


    """
    if detect_single and df[by].nunique() == 1:
        return func(df)
    else:
        labels, idfs = zip(*df.groupby(by))
        return lazy_show_over_data(idfs, func, labels=labels, style=style)
