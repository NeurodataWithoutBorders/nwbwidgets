from collections import OrderedDict

import h5py
import hdmf
import pynwb

from ipywidgets import widgets

import zarr
import ndx_grayscalevolume
from ndx_icephys_meta.icephys import SweepSequences
from ndx_spectrum import Spectrum

from .dynamictablesummary import DynamicTableSummaryWidget

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
            "Summary": DynamicTableSummaryWidget,
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
    #pynwb.ophys.RoiResponseSeries: ophys.RoiResponseSeriesWidget,
    pynwb.ophys.RoiResponseSeries: timeseries.TrializedTimeSeries,
    # pynwb.ophys.RoiResponseSeries: OrderedDict(
    #     {
    #     "traces": ophys.RoiResponseSeriesWidget,
    #     #"trial_aligned": timeseries.TrializedTimeSeries,
    #     }
    # ),
    pynwb.misc.AnnotationSeries: OrderedDict(
        {
            "text": base.show_text_fields,
            "times": misc.show_annotations
        }
    ),
    pynwb.core.LabelledDict: base.dict2accordion,
    pynwb.ProcessingModule: base.processing_module,
    hdmf.common.DynamicTable:
        {
        "Summary": DynamicTableSummaryWidget,
        "table": show_dynamic_table,
        },
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
    pynwb.icephys.SequentialRecordingsTable: {
        "Summary": DynamicTableSummaryWidget,
        "table": show_dynamic_table,
        "I-V Analysis": icephys.IVCurveWidget
    }
}


def nwb2widget(node, neurodata_vis_spec=default_neurodata_vis_spec):
    return base.nwb2widget(node, neurodata_vis_spec)