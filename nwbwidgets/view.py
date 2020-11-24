from collections import OrderedDict

import h5py
import hdmf
import ndx_grayscalevolume
import pynwb
import zarr
from ipywidgets import widgets
from ndx_icephys_meta.icephys import SweepSequences
from ndx_spectrum import Spectrum

from nwbwidgets import behavior, misc, base, ecephys, image, ophys, icephys, timeseries, file, placefield, spectrum


# def show_dynamic_table(node: DynamicTable, **kwargs):
def show_dynamic_table(node, **kwargs) -> widgets.Widget:
    if node.name == 'electrodes':
        return ecephys.show_electrodes(node)
    return base.render_dataframe(node)


default_neurodata_vis_spec = {
    pynwb.NWBFile: file.show_nwbfile,
    SweepSequences: icephys.show_sweep_sequences,
    pynwb.behavior.BehavioralEvents: behavior.show_behavioral_events,
    pynwb.ecephys.LFP: ecephys.show_lfp,
    pynwb.misc.Units: OrderedDict({
        'Session Raster': misc.RasterWidget,
        'Grouped PSTH': misc.PSTHWidget,
        'Raster Grid': misc.RasterGridWidget,
        'table': show_dynamic_table}),
    pynwb.misc.DecompositionSeries: misc.show_decomposition_series,
    pynwb.file.Subject: base.show_fields,
    pynwb.ecephys.SpikeEventSeries: ecephys.show_spike_event_series,
    pynwb.ophys.ImageSegmentation: ophys.show_image_segmentation,
    pynwb.ophys.TwoPhotonSeries: ophys.TwoPhotonSeriesWidget,
    ndx_grayscalevolume.GrayscaleVolume: ophys.show_grayscale_volume,
    pynwb.ophys.PlaneSegmentation: ophys.route_plane_segmentation,
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
        'trace': behavior.plotly_show_spatial_trace,
        'rate map': placefield.route_placefield,
        '1D rate map': placefield.PlaceField1DWidget}),
    pynwb.image.GrayscaleImage: image.show_grayscale_image,
    pynwb.image.RGBImage: image.show_rbga_image,
    pynwb.image.RGBAImage: image.show_rbga_image,
    pynwb.base.Image: image.show_rbga_image,
    pynwb.image.ImageSeries: image.show_image_series,
    pynwb.image.IndexSeries: image.show_index_series,
    pynwb.TimeSeries: timeseries.show_timeseries,
    pynwb.core.NWBContainer: base.show_neurodata_base,
    pynwb.core.NWBDataInterface: base.show_neurodata_base,
    h5py.Dataset: base.show_dset,
    zarr.core.Array: base.show_dset,
    Spectrum: spectrum.show_spectrum,
    pynwb.behavior.CompassDirection: behavior.show_position
}


def nwb2widget(node, neurodata_vis_spec=default_neurodata_vis_spec):
    return base.nwb2widget(node, neurodata_vis_spec)
