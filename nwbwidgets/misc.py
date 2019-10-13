import matplotlib.pyplot as plt
import numpy as np
from pynwb.misc import AnnotationSeries
from ipywidgets import widgets


def show_annotations(annotations: AnnotationSeries, **kwargs):
    default_kwargs = {'marker': "|", 'linestyle': ''}
    for key, val in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = val

    fig, ax = plt.subplots()
    ax.plot(annotations.timestamps, np.ones(len(annotations.timestamps)), **kwargs)
    ax.set_xlabel('time (s)')
    return fig


def show_units(node, **kwargs):
    field_lay = widgets.Layout(max_height='40px', max_width='700px',
                               min_height='30px', min_width='120px')
    info = []
    for col in node.colnames:
        lbl_key = widgets.Label(col+':', layout=field_lay)
        lbl_val = widgets.Label(str(node[col][0]), layout=field_lay)
        info.append(widgets.HBox(children=[lbl_key, lbl_val]))
    vbox0 = widgets.VBox(info)

    unit = widgets.BoundedIntText(value=0, min=0, max=node.columns[0][:].shape[0]-1,
                                  description='Unit')
    def update_x_range(change):
        for ind, ch in enumerate(vbox0.children):
            ch.children[1].value = str(node[node.colnames[ind]][change.new])
    unit.observe(update_x_range, 'value')

    vbox1 = widgets.VBox([unit, vbox0])
    return vbox1
