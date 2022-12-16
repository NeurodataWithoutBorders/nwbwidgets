from collections import OrderedDict

import h5py
import hdmf
import ndx_grayscalevolume
import pynwb
import zarr
from ipywidgets import widgets
from ndx_icephys_meta.icephys import SweepSequences
from ndx_spectrum import Spectrum

from .base import nwb2widget as nwb2widget_base
from .base import (
    dict2accordion,
    processing_module,
    render_dataframe,
    show_dset,
    show_fields,
    show_multi_container_interface,
    show_neurodata_base,
    show_text_fields,
)

from .behavior import route_spatial_series, show_behavioral_events
from .dynamictablesummary import DynamicTableSummaryWidget
from .ecephys import ElectricalSeriesWidget, show_electrodes, show_spike_event_series
from .file import show_nwbfile
from .icephys import IVCurveWidget, show_sweep_sequences
from .image import (
    ImageSeriesWidget,
    show_grayscale_image,
    show_index_series,
    show_rbga_image,
)
from .misc import (
    PSTHWidget,
    RasterGridWidget,
    RasterWidget,
    TuningCurveExtendedWidget,
    TuningCurveWidget,
    show_annotations,
    show_decomposition_series,
)
from .ophys import (
    RoiResponseSeriesWidget,
    TwoPhotonSeriesWidget,
    route_plane_segmentation,
    show_df_over_f,
    show_grayscale_volume,
    show_image_segmentation,
)
from .spectrum import show_spectrum
from .timeseries import route_trialized_time_series, show_timeseries


# def show_dynamic_table(node: DynamicTable, **kwargs):
def show_dynamic_table(node, **kwargs) -> widgets.Widget:
    if node.name == "electrodes":
        return show_electrodes(node)
    return render_dataframe(node)


default_neurodata_vis_spec = {
    pynwb.NWBFile: show_nwbfile,
    SweepSequences: show_sweep_sequences,
    pynwb.behavior.BehavioralEvents: show_behavioral_events,
    pynwb.misc.Units: OrderedDict(
        {
            "Summary": DynamicTableSummaryWidget,
            "Session Raster": RasterWidget,
            "Grouped PSTH": PSTHWidget,
            "Raster Grid": RasterGridWidget,
            "Tuning Curves": TuningCurveWidget,
            "Combined": TuningCurveExtendedWidget,
            "table": show_dynamic_table,
        }
    ),
    pynwb.misc.DecompositionSeries: show_decomposition_series,
    pynwb.file.Subject: show_fields,
    pynwb.ecephys.SpikeEventSeries: show_spike_event_series,
    pynwb.ophys.ImageSegmentation: show_image_segmentation,
    pynwb.ophys.TwoPhotonSeries: TwoPhotonSeriesWidget,
    ndx_grayscalevolume.GrayscaleVolume: show_grayscale_volume,
    pynwb.ophys.PlaneSegmentation: route_plane_segmentation,
    pynwb.ophys.DfOverF: show_df_over_f,
    pynwb.ophys.RoiResponseSeries: OrderedDict(
        {
            "trial_aligned": route_trialized_time_series,
            "traces": RoiResponseSeriesWidget,
        }
    ),
    pynwb.misc.AnnotationSeries: OrderedDict({"text": show_text_fields, "times": show_annotations}),
    pynwb.core.LabelledDict: dict2accordion,
    pynwb.ProcessingModule: processing_module,
    hdmf.common.DynamicTable: {
        "Summary": DynamicTableSummaryWidget,
        "table": show_dynamic_table,
    },
    pynwb.ecephys.ElectricalSeries: ElectricalSeriesWidget,
    pynwb.behavior.SpatialSeries: route_spatial_series,
    pynwb.image.GrayscaleImage: show_grayscale_image,
    pynwb.image.RGBImage: show_rbga_image,
    pynwb.image.RGBAImage: show_rbga_image,
    pynwb.base.Image: show_rbga_image,
    pynwb.image.ImageSeries: ImageSeriesWidget,
    pynwb.image.IndexSeries: show_index_series,
    pynwb.TimeSeries: show_timeseries,
    pynwb.core.MultiContainerInterface: show_multi_container_interface,
    pynwb.core.NWBContainer: show_neurodata_base,
    pynwb.core.NWBDataInterface: show_neurodata_base,
    h5py.Dataset: show_dset,
    zarr.core.Array: show_dset,
    Spectrum: show_spectrum,
    pynwb.icephys.SequentialRecordingsTable: {
        "Summary": DynamicTableSummaryWidget,
        "table": show_dynamic_table,
        "I-V Analysis": IVCurveWidget,
    },
}


def nwb2widget(node, neurodata_vis_spec=default_neurodata_vis_spec):
    return nwb2widget_base(node, neurodata_vis_spec)
