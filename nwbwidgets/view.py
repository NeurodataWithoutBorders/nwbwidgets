import matplotlib.pyplot as plt
import ipywidgets as widgets
import pynwb
from collections import OrderedDict
from nwbwidgets import behavior, misc, base, ecephys, image


def fig2widget(fig):
    out = widgets.Output()
    with out:
        plt.show(fig)
    return out


def dict2accordion(d):
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
            children[change.new] = nwb2widget(list(d.values())[change.new])
            change.owner.children = children

    accordion.observe(on_selected_index, names='selected_index')

    return accordion


def show_text_fields(node, exclude=('comments', 'interval')):
    info = []
    for key in node.fields:
        if key not in exclude and isinstance(key, (str, float, int)):
            info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    return widgets.VBox(info)


neurodata_vis_spec = OrderedDict({
    pynwb.misc.AnnotationSeries: OrderedDict({
        'text': show_text_fields,
        'times': misc.show_annotations}),
    pynwb.core.LabelledDict: dict2accordion,
    pynwb.ProcessingModule: lambda x: nwb2widget(x.data_interfaces),
    pynwb.core.DynamicTable: base.show_dynamic_table,
    pynwb.ecephys.LFP: ecephys.show_lfp,
    pynwb.behavior.Position: behavior.show_position,
    pynwb.behavior.SpatialSeries: behavior.show_spatial_series,
    pynwb.image.GrayscaleImage: image.show_grayscale_image,
    pynwb.image.IndexSeries: image.show_index_series,
    pynwb.TimeSeries: base.show_timeseries,
    pynwb.core.NWBBaseType: base.show_neurodata_base
})


def vis2widget(vis):
    if isinstance(vis, widgets.Widget):
        return vis
    elif isinstance(vis, plt.Figure):
        return fig2widget(vis)
    else:
        raise ValueError('unsupported vis type')


def nwb2widget(node,  neurodata_vis_spec=neurodata_vis_spec):

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
                return vis2widget(spec(node))
    out1 = widgets.Output()
    with out1:
        print(node)
    return out1

