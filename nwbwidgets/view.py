import pynwb
import ndx_grayscalevolume
from collections import OrderedDict
from nwbwidgets import behavior, misc, base, ecephys, image, ophys
import hdmf


default_neurodata_vis_spec = {
    pynwb.ecephys.LFP: ecephys.show_lfp,
    pynwb.misc.Units: OrderedDict({
        'table': base.show_dynamic_table,
        'raster': misc.raster_widget}),
    pynwb.misc.DecompositionSeries: misc.show_decomposition_series,
    pynwb.file.Subject: base.show_subject,
    pynwb.ecephys.SpikeEventSeries: ecephys.show_spike_event_series,
    pynwb.ophys.ImageSegmentation: ophys.show_image_segmentation,
    pynwb.ophys.TwoPhotonSeries: ophys.show_two_photon_series,
    ndx_grayscalevolume.GrayscaleVolume: ophys.show_grayscale_volume,
    pynwb.ophys.PlaneSegmentation: ophys.show_plane_segmentation,
    pynwb.ophys.DfOverF: ophys.show_df_over_f,
    pynwb.ophys.RoiResponseSeries: ophys.show_roi_response_series,
    pynwb.misc.AnnotationSeries: OrderedDict({
        'text': base.show_text_fields,
        'times': misc.show_annotations}),
    pynwb.core.LabelledDict: base.dict2accordion,
    pynwb.ProcessingModule: base.processing_module,
    hdmf.common.DynamicTable: base.show_dynamic_table,
    pynwb.ecephys.ElectricalSeries: OrderedDict({
        'Fields': base.show_ts_fields,
        'Traces': ecephys.show_voltage_traces,
    }),
    pynwb.behavior.Position: behavior.show_position,
    pynwb.behavior.SpatialSeries: OrderedDict({
        'over time': behavior.show_spatial_series_over_time,
        'trace': behavior.show_spatial_series}),
    pynwb.image.GrayscaleImage: image.show_grayscale_image,
    pynwb.image.ImageSeries: image.show_image_series,
    pynwb.image.IndexSeries: image.show_index_series,
    pynwb.TimeSeries: base.show_timeseries,
    pynwb.core.NWBDataInterface: base.show_neurodata_base,
}


def nwb2widget(node,  neurodata_vis_spec=default_neurodata_vis_spec):
    return base.nwb2widget(node, neurodata_vis_spec)
