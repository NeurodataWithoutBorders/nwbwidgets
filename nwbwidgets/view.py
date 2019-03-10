import matplotlib.pyplot as plt
import ipywidgets as widgets
import pynwb
from nwbwidgets import behavior, misc, base, ecephys


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
            children[change.new] = nwb2widget(list(d.values())[change.new], neurodata_vis)
            change.owner.children = children

    accordion.observe(on_selected_index, names='selected_index')

    return accordion


def show_text_fields(node, exclude=('comments', 'interval')):
    info = []
    for key in node.fields:
        if key not in exclude and isinstance(key, (str, float, int)):
            info.append(widgets.Text(value=repr(getattr(node, key)), description=key, disabled=True))
    return widgets.VBox(info)


neurodata_vis = (
    (
        pynwb.misc.AnnotationSeries, (
            ('text', show_text_fields),
            ('times', misc.show_annotations))
    ),
    (pynwb.core.LabelledDict, dict2accordion),
    (pynwb.ProcessingModule, lambda x: nwb2widget(x.data_interfaces)),
    (pynwb.core.DynamicTable, base.show_dynamic_table),
    (pynwb.ecephys.LFP, ecephys.show_lfp),
    (pynwb.behavior.Position, behavior.show_position),
    (pynwb.behavior.SpatialSeries, behavior.show_spatial_series),
    (pynwb.TimeSeries, base.show_timeseries),
    (pynwb.core.NWBBaseType, base.show_neurodata_base)
)


def vis2widget(vis):
    if isinstance(vis, widgets.Widget):
        return vis
    elif isinstance(vis, plt.Figure):
        return fig2widget(vis)
    else:
        raise ValueError('unsupported vis type')


def nwb2widget(node,  neurodata_vis=neurodata_vis):

    for ndtype, vis_funcs in neurodata_vis:
        if isinstance(node, ndtype):
            if isinstance(vis_funcs, tuple):
                children = [vis_funcs[0][1](node)] + [widgets.HTML('Rendering...')
                                                      for _ in range(len(vis_funcs) - 1)]
                tab = widgets.Tab(children=children)
                [tab.set_title(i, label) for i, (label, _) in enumerate(vis_funcs)]

                def on_selected_index(change):
                    if isinstance(change.owner.children[change.new], widgets.HTML):
                        children[change.new] = vis2widget(vis_funcs[change.new][1](node))
                        change.owner.children = children

                tab.observe(on_selected_index, names='selected_index')

                return tab
            else:
                return vis2widget(vis_funcs(node))
    out1 = widgets.Output()
    with out1:
        print(node)
    return out1

