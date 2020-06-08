from nwbwidgets import view
from ipywidgets import widgets
from collections.abc import Iterable
from datetime import datetime
from pynwb.file import NWBFile
from .base import lazy_show_over_data
from functools import partial


def show_nwbfile(nwbfile: NWBFile, neurodata_vis_spec: dict) -> widgets.Widget:
    """
        Gets a pynwb object and returns a Vertical Box containing textual info and
        an expandable Accordion with it's children.
    """
    field_lay = widgets.Layout(max_height='40px', max_width='500px',
                               min_height='30px', min_width='180px')
    info = []  # string data type, exposed as a Text widget
    neuro_data = []  # more complex data types, also with children
    labels = []
    for key, value in nwbfile.fields.items():
        if isinstance(value, (str, datetime)):
            lbl_key = widgets.Label(key + ':', layout=field_lay)
            lbl_val = widgets.Label(str(value), layout=field_lay)
            info.append(widgets.HBox(children=[lbl_key, lbl_val]))
        elif key == 'related_publications':
            pub_list = []
            for pub in value:
                if isinstance(pub, bytes):
                    pub = pub.decode()
                pub_list.append(widgets.HTML(value="<a href=http://dx.doi.org/" + pub[4:] + ">" + pub + "</a>"))
            lbl_key = widgets.Label(key + ':', layout=field_lay)
            pub_list.insert(0, lbl_key)
            info.append(widgets.HBox(children=pub_list))
        elif key == 'experimenter':
            lbl_experimenter = widgets.Label('Experimenter:', layout=field_lay)
            if isinstance(value, (list, tuple)):
                if isinstance(value[0], str):
                    lbl_names = widgets.Label(', '.join(value), layout=field_lay)
                elif isinstance(value[0], bytes):
                    lbl_names = widgets.Label(b', '.join(value).decode(), layout=field_lay)
                else:
                    raise ValueError('unrecognized type for experimenter')
            else:
                lbl_names = widgets.Label(value, layout=field_lay)
            hbox_exp = widgets.HBox(children=[lbl_experimenter, lbl_names])
            info.append(hbox_exp)
        elif (isinstance(value, Iterable) and len(value)) or value:
            neuro_data.append(value)
            if hasattr(nwbfile.fields[key], 'description') and nwbfile.fields[key].description:
                labels.append(key + ': ' + nwbfile.fields[key].description)
            else:
                labels.append(key)
    func_ = partial(view.nwb2widget, neurodata_vis_spec=neurodata_vis_spec)
    accordion = lazy_show_over_data(neuro_data, func_, labels=labels, style=widgets.Accordion)

    return widgets.VBox(info + [accordion])