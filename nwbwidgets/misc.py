import matplotlib.pyplot as plt
import numpy as np
from pynwb.misc import AnnotationSeries
from ipywidgets import widgets
import bottleneck as bn
from nwbwidgets import view


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
                                  description='Unit', layout=field_lay)
    def update_x_range(change):
        for ind, ch in enumerate(vbox0.children):
            ch.children[1].value = str(node[node.colnames[ind]][change.new])
    unit.observe(update_x_range, 'value')

    vbox1 = widgets.VBox([unit, vbox0])

    ntabs = 2
    children = [widgets.HTML('Rendering...') for _ in range(ntabs)]
    children[0] = vbox1

    def on_selected_index(change):
        # Click on Traces Tab
        if change.new == 1 and isinstance(change.owner.children[1], widgets.HTML):
            widget_box = show_unit_traces(node)
            children[1] = widget_box
            change.owner.children = children

    tab_nest = widgets.Tab()
    tab_nest.children = children
    tab_nest.set_title(0, 'Fields')
    tab_nest.set_title(1, 'Traces')
    tab_nest.observe(on_selected_index, names='selected_index')
    return tab_nest


def show_unit_traces(node):
    def control_plot(unit, x0, x1, window):
        xx = np.arange(x0, x1+1)
        fig, ax = plt.subplots(figsize=(18, 10))
        spkt = (node['spike_times'][unit][:]*1000).astype('int')
        binned = np.zeros(len(xx))
        spkt1 = spkt[spkt>x0]
        spkt1 = spkt1[spkt1<x1]
        binned[spkt1-x0] = 1
        # Calculates moving average from binned spike times
        smoothed = bn.move_mean(binned, window=window, min_count=100)
        peak = np.nanmax(1000*smoothed)
        ax.plot(xx, binned*peak, color='lightgrey')
        ax.plot(xx, 1000*smoothed, color='k')
        ax.set_xlabel('Time [ms]', fontsize=20)
        ax.set_ylabel('Rate [spks/sec]', fontsize=20)
        ax.set_xlim([x0, x1])
        ax.set_ylim([0, peak*1.1])
        plt.show()
        return view.fig2widget(fig)

    field_lay = widgets.Layout(max_height='40px', max_width='100px',
                               min_height='30px', min_width='50px')
    lbl_unit = widgets.Label('Unit:', layout=field_lay)
    unit1 = widgets.BoundedIntText(value=0, min=0, max=node.columns[0][:].shape[0]-1,
                                   layout=field_lay)
    lbl_blank0 = widgets.Label('       ', layout=field_lay)
    lbl_time = widgets.Label('Time [ms]:', layout=field_lay)

    tEnd = (node['spike_times'][0][-1]*1000 + 1).astype('int')
    x0 = widgets.BoundedIntText(value=0, min=0, max=tEnd-100,
                                layout=field_lay)
    x1 = widgets.BoundedIntText(value=10000, min=100, max=tEnd,
                                layout=field_lay)
    lbl_blank1 = widgets.Label('       ', layout=field_lay)
    lbl_window = widgets.Label('Window [ms]:', layout=field_lay)
    window = widgets.BoundedIntText(value=1000, min=10, max=20000,
                                    layout=field_lay)
    hbox0 = widgets.HBox(children=[lbl_unit, unit1, lbl_blank0, lbl_time,
                                   x0, x1, lbl_blank1, lbl_window, window])
    controls = {
        'unit': unit1,
        'x0': x0,
        'x1': x1,
        'window': window
    }
    out_fig = widgets.interactive_output(control_plot, controls)
    vbox = widgets.VBox([hbox0, out_fig])
    return vbox
