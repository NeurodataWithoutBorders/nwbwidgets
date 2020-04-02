import pynwb
import ndx_grayscalevolume
from collections import OrderedDict
from nwbwidgets import behavior, misc, base, ecephys, image, ophys, icephys, timeseries, file
import hdmf
from functools import partial
from ndx_icephys_meta.icephys import SweepSequences


default_neurodata_vis_spec = {
    pynwb.NWBFile: file.show_nwbfile,
    SweepSequences: icephys.show_sweep_sequences,
    pynwb.behavior.BehavioralEvents: behavior.show_behavioral_events,
    pynwb.ecephys.LFP: ecephys.show_lfp,
    pynwb.misc.Units: OrderedDict({
        'Session Raster': misc.RasterWidget,
        'Grouped PSTH': misc.PSTHWidget,
        'Raster Grid': misc.RasterGridWidget,
        'table': base.show_dynamic_table}),
    pynwb.misc.DecompositionSeries: misc.show_decomposition_series,
    pynwb.file.Subject: base.show_fields,
    pynwb.ophys.ImagingPlane: base.show_fields,
    pynwb.ecephys.SpikeEventSeries: ecephys.show_spike_event_series,
    pynwb.ophys.ImageSegmentation: ophys.show_image_segmentation,
    pynwb.ophys.TwoPhotonSeries: ophys.show_two_photon_series,
    ndx_grayscalevolume.GrayscaleVolume: ophys.show_grayscale_volume,
    pynwb.ophys.PlaneSegmentation: ophys.show_plane_segmentation,
    pynwb.ophys.DfOverF: ophys.show_df_over_f,
    pynwb.ophys.RoiResponseSeries: timeseries.traces_widget,
    pynwb.misc.AnnotationSeries: OrderedDict({
        'text': base.show_text_fields,
        'times': misc.show_annotations}),
    pynwb.core.LabelledDict: base.dict2accordion,
    pynwb.ProcessingModule: base.processing_module,
    hdmf.common.DynamicTable: base.show_dynamic_table,
    pynwb.ecephys.ElectricalSeries: OrderedDict({
        'Fields': timeseries.show_ts_fields,
        'Traces': partial(timeseries.traces_widget,
                          start=0, dur=10,
                          trace_starting_range=(0, 5)),
    }),
    pynwb.behavior.Position: behavior.show_position,
    pynwb.behavior.SpatialSeries: OrderedDict({
        'over time': behavior.show_spatial_series_over_time,
        'trace': behavior.show_spatial_series}),
    pynwb.image.GrayscaleImage: image.show_grayscale_image,
    pynwb.image.RGBImage: image.show_rbg_image,
    pynwb.image.ImageSeries: image.show_image_series,
    pynwb.image.IndexSeries: image.show_index_series,
    pynwb.TimeSeries: timeseries.show_timeseries,
    pynwb.core.NWBDataInterface: base.show_neurodata_base,
}


def nwb2widget(node,  neurodata_vis_spec=default_neurodata_vis_spec):
    return base.nwb2widget(node, neurodata_vis_spec)
