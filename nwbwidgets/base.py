from nwbwidgets import view
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython import display
from collections import Iterable
from pynwb import TimeSeries, ProcessingModule
from pynwb.core import NWBDataInterface
from collections import OrderedDict
from hdmf.common import DynamicTable
from matplotlib.pyplot import Figure
from datetime import datetime


def show_timeseries(node: TimeSeries, **kwargs):
    info = []
    for key in ('description', 'comments', 'unit', 'resolution', 'conversion'):
        info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    children = [widgets.VBox(info)]

    fig, ax = plt.subplots()
    if node.timestamps:
        ax.plot(node.timestamps, node.data)
    else:
        #ax.plot(np.arange(len(node.data)) / node.rate + node.starting_time, node.data, **kwargs)
        ax.plot(np.arange(len(node.data)) / node.rate + node.starting_time, node.data)
    ax.set_xlabel('time (s)')
    if node.unit:
        ax.set_ylabel(node.unit)

    children.append(fig2widget(fig))

    return widgets.HBox(children=children)


def show_subject(node, **kwargs):
    field_lay = widgets.Layout(max_height='40px', max_width='150px',
                               min_height='30px', min_width='70px')
    info = []
    for key, val in node.fields.items():
        lbl_key = widgets.Label(key+':', layout=field_lay)
        lbl_val = widgets.Label(str(val), layout=field_lay)
        info.append(widgets.HBox(children=[lbl_key, lbl_val]))
    vbox = widgets.VBox(info)
    return vbox


# def show_dynamic_table(node: DynamicTable, **kwargs):
def show_dynamic_table(node, **kwargs):
    out1 = widgets.Output()
    with out1:
        display.display(node.to_dataframe())
    return out1


def show_neurodata_base(node: NWBDataInterface, neurodata_vis_spec: OrderedDict):
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
        if isinstance(value, str):
            #info.append(widgets.Text(value=repr(value), description=key, disabled=True))
            lbl_key = widgets.Label(key+':', layout=field_lay)
            lbl_val = widgets.Label(value, layout=field_lay)
            info.append(widgets.HBox(children=[lbl_key, lbl_val]))
        elif isinstance(value, datetime):
            #info.append(widgets.Text(value=str(value), description=key, disabled=True))
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
        elif key == 'keywords':
            lbl_keywords = widgets.Label('Keywords:', layout=field_lay)
            kwds = ', '.join([kw for kw in value])
            lbl_kwdnames = widgets.Label(kwds, layout=field_lay)
            hbox_kwd = widgets.HBox(children=[lbl_keywords, lbl_kwdnames])
            info.append(hbox_kwd)
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


def dict2accordion(d, neurodata_vis_spec, **pass_kwargs):
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


def nwb2widget(node,  neurodata_vis_spec, **pass_kwargs):

    for ndtype, spec in neurodata_vis_spec.items():
        if isinstance(node, ndtype):
            if isinstance(spec, (dict, OrderedDict)):
                tabs_spec = list(spec.items())

                children = [tabs_spec[0][1](node)] + [widgets.HTML('Rendering...')
                                                      for _ in range(len(tabs_spec) - 1)]
                tab = widgets.Tab(children=children)
                [tab.set_title(i, label) for i, (label, _) in enumerate(tabs_spec)]

                def on_selected_index(change):
                    if isinstance(change.owner.children[change.new], widgets.HTML):
                        children[change.new] = vis2widget(tabs_spec[change.new][1](node))
                        change.owner.children = children

                tab.observe(on_selected_index, names='selected_index')

                return tab
            elif callable(spec):
                return vis2widget(spec(node, neurodata_vis_spec=neurodata_vis_spec, **pass_kwargs))
    out1 = widgets.Output()
    with out1:
        print(node)
    return out1


def vis2widget(vis):
    if isinstance(vis, widgets.Widget):
        return vis
    elif isinstance(vis, plt.Figure):
        return fig2widget(vis)
    else:
        raise ValueError('unsupported vis type')


def fig2widget(fig: Figure, **kwargs):
    out = widgets.Output()
    with out:
        plt.show(fig)
    return out


def processing_module(node: ProcessingModule, neurodata_vis_spec: OrderedDict):
    return nwb2widget(node.data_interfaces, neurodata_vis_spec=neurodata_vis_spec)


def show_text_fields(node, exclude=('comments', 'interval'), **kwargs):
    info = []
    for key in node.fields:
        if key not in exclude and isinstance(key, (str, float, int)):
            info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    return widgets.VBox(info)
