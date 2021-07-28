from collections import OrderedDict
from collections.abc import Iterable

import h5py
import hdmf
import ndx_grayscalevolume
import pynwb
import zarr
from ipywidgets import widgets
from ndx_icephys_meta.icephys import SweepSequences
from ndx_spectrum import Spectrum

from nwbwidgets import (
    behavior,
    misc,
    base,
    ecephys,
    image,
    ophys,
    icephys,
    timeseries,
    file,
    spectrum,
)


# def show_dynamic_table(node: DynamicTable, **kwargs):
def show_dynamic_table(node, **kwargs) -> widgets.Widget:
    if node.name == "electrodes":
        return ecephys.show_electrodes(node)
    return base.render_dataframe(node)


default_neurodata_vis_spec = {
    pynwb.NWBFile: file.show_nwbfile,
    SweepSequences: icephys.show_sweep_sequences,
    pynwb.behavior.BehavioralEvents: behavior.show_behavioral_events,
    pynwb.misc.Units: OrderedDict(
        {
            "Session Raster": misc.RasterWidget,
            "Grouped PSTH": misc.PSTHWidget,
            "Raster Grid": misc.RasterGridWidget,
            "Tuning Curves": misc.TuningCurveWidget,
            "Combined": misc.TuningCurveExtendedWidget,
            "table": show_dynamic_table,
        }
    ),
    pynwb.misc.DecompositionSeries: misc.show_decomposition_series,
    pynwb.file.Subject: base.show_fields,
    pynwb.ecephys.SpikeEventSeries: ecephys.show_spike_event_series,
    pynwb.ophys.ImageSegmentation: ophys.show_image_segmentation,
    pynwb.ophys.TwoPhotonSeries: ophys.TwoPhotonSeriesWidget,
    ndx_grayscalevolume.GrayscaleVolume: ophys.show_grayscale_volume,
    pynwb.ophys.PlaneSegmentation: ophys.route_plane_segmentation,
    pynwb.ophys.DfOverF: ophys.show_df_over_f,
    pynwb.ophys.RoiResponseSeries: ophys.RoiResponseSeriesWidget,
    pynwb.misc.AnnotationSeries: OrderedDict(
        {
            "text": base.show_text_fields, 
            "times": misc.show_annotations
        }
    ),
    pynwb.core.LabelledDict: base.dict2accordion,
    pynwb.ProcessingModule: base.processing_module,
    hdmf.common.DynamicTable: show_dynamic_table,
    pynwb.ecephys.ElectricalSeries: ecephys.ElectricalSeriesWidget,
    pynwb.behavior.SpatialSeries: behavior.route_spatial_series,
    pynwb.image.GrayscaleImage: image.show_grayscale_image,
    pynwb.image.RGBImage: image.show_rbga_image,
    pynwb.image.RGBAImage: image.show_rbga_image,
    pynwb.base.Image: image.show_rbga_image,
    pynwb.image.ImageSeries: image.ImageSeriesWidget,
    pynwb.image.IndexSeries: image.show_index_series,
    pynwb.TimeSeries: timeseries.show_timeseries,
    pynwb.core.MultiContainerInterface: base.show_multi_container_interface,
    pynwb.core.NWBContainer: base.show_neurodata_base,
    pynwb.core.NWBDataInterface: base.show_neurodata_base,
    h5py.Dataset: base.show_dset,
    zarr.core.Array: base.show_dset,
    Spectrum: spectrum.show_spectrum,
}


def nwb2widget(node, neurodata_vis_spec=default_neurodata_vis_spec, include_widgets: list=None):
    
    # Include user custom widgets
    if include_widgets is None:
        include_widgets = list()
    for item in include_widgets:
        widget = item.get('widget', None)
        label = item.get('label', None)
        pynwb_class = item.get('pynwb_class', None)
        if widget and label and pynwb_class:
            neurodata_vis_spec = include_widget(
                neurodata_vis_spec=neurodata_vis_spec,
                widget=widget, 
                label=label, 
                pynwb_class=pynwb_class
            )
        else:
            print(f'Failed to include widget {widget} with label {label} to pynwb_class {pynwb_class}')

    return base.nwb2widget(node, neurodata_vis_spec)


def include_widget(neurodata_vis_spec, widget, label, pynwb_class):
    if pynwb_class in neurodata_vis_spec:
        neurodata_vis_spec[pynwb_class].update({label: widget})
    else:
        neurodata_vis_spec[pynwb_class] = {label: widget}
    return neurodata_vis_spec
