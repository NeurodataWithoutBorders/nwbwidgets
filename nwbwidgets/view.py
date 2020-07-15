import pynwb
import ndx_grayscalevolume
from collections import OrderedDict
from nwbwidgets import behavior, misc, base, ecephys, image, ophys, icephys, timeseries, file
import hdmf
from ndx_icephys_meta.icephys import SweepSequences
from ipywidgets import widgets
from .ecephys import ElectrodesWidget
from .base import render_dataframe
import h5py
import zarr


# def show_dynamic_table(node: DynamicTable, **kwargs):
def show_dynamic_table(node, **kwargs) -> widgets.Widget:
    if node.name == 'electrodes':
        return ElectrodesWidget(node)
    return render_dataframe(node)


default_neurodata_vis_spec = {
    pynwb.NWBFile: file.show_nwbfile,
    SweepSequences: icephys.show_sweep_sequences,
    pynwb.behavior.BehavioralEvents: behavior.show_behavioral_events,
    pynwb.ecephys.LFP: ecephys.show_lfp,
    pynwb.misc.Units: OrderedDict({
        'Session Raster': misc.RasterWidgetPlotly,
        'Grouped PSTH': misc.PSTHWidget,
        'Raster Grid': misc.RasterGridWidget,
        'table': show_dynamic_table}),
    pynwb.misc.DecompositionSeries: misc.show_decomposition_series,
    pynwb.file.Subject: base.show_fields,
    pynwb.ophys.ImagingPlane: base.show_fields,
    pynwb.ecephys.SpikeEventSeries: ecephys.show_spike_event_series,
    pynwb.ophys.ImageSegmentation: ophys.show_image_segmentation,
    pynwb.ophys.TwoPhotonSeries: ophys.show_two_photon_series,
    ndx_grayscalevolume.GrayscaleVolume: ophys.show_grayscale_volume,
    pynwb.ophys.PlaneSegmentation: ophys.show_plane_segmentation,
    pynwb.ophys.DfOverF: ophys.show_df_over_f,
    pynwb.ophys.RoiResponseSeries: ophys.RoiResponseSeriesWidget,
    pynwb.misc.AnnotationSeries: OrderedDict({
        'text': base.show_text_fields,
        'times': misc.show_annotations}),
    pynwb.core.LabelledDict: base.dict2accordion,
    pynwb.ProcessingModule: base.processing_module,
    hdmf.common.DynamicTable: show_dynamic_table,
    pynwb.ecephys.ElectricalSeries: ecephys.ElectricalSeriesWidget,
    pynwb.behavior.Position: behavior.show_position,
    pynwb.behavior.SpatialSeries: OrderedDict({
        'over time': timeseries.SeparateTracesPlotlyWidget,
        'trace': behavior.plotly_show_spatial_trace}),
    pynwb.image.GrayscaleImage: image.show_grayscale_image,
    pynwb.image.RGBImage: image.show_rbg_image,
    pynwb.image.ImageSeries: image.show_image_series,
    pynwb.image.IndexSeries: image.show_index_series,
    pynwb.TimeSeries: timeseries.show_timeseries,
    pynwb.core.NWBDataInterface: base.show_neurodata_base,
    h5py.Dataset: base.show_dset,
    zarr.core.Array: base.show_dset
}


def nwb2widget(node,  neurodata_vis_spec=default_neurodata_vis_spec):
    return base.nwb2widget(node, neurodata_vis_spec)
