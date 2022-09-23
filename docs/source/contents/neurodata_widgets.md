# Neurodata widgets

## Session metadata
Experimental sessions metadata is stored as `pynwb.NWBFile` object in NWB files and is rendered as a widget by `nwbwidgets.file.show_nwbfile` function.

(EXAMPLE_GIF)

## Subject metadata
Subejct's metadata is stored as `pynwb.file.Subject` object in NWB files and is rendered as a widget by `nwbwidgets.base.show_fields` function.

(EXAMPLE_GIF)

## Extracellular electrophysiology series
Ecephys series are stored as `pynwb.ecephys.ElectricalSeries` objects in NWB files and are rendered as a widget by `nwbwidgets.ecephys.ElectricalSeriesWidget`.

(EXAMPLE_GIF)

## Units
Units spiking activity is stored as `pynwb.misc.Units` objects in NWB files and can be rendered as multiple widgets: `nwbwidgets.dynamictablesummary.DynamicTableSummaryWidget`, `nwbwidgets.misc.RasterWidget`, `nwbwidgets.misc.PSTHWidget`, `nwbwidgets.misc.RasterGridWidget`, `nwbwidgets.misc.TuningCurveWidget` and `nwbwidgets.misc.TuningCurveExtendedWidget`.

(EXAMPLE_GIF)

