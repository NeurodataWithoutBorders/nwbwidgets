# Upcoming

# v0.10.2
* Prevented the display of video assets on DANDI from the `Panel` dropdown. [PR #281](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/281)
* Remove `trials` from the accordion of `nwb2widget` (it will display in the `intervals` tab alongside any other `TimeIntervals`). [PR #281](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/281)
* Prevent the `ElectrodeGroupWidget` from loading if positions (specifically, `x`) are missing in conjunction with nwb-schema versions that allow those columns to be optional. [PR #280](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/280)



# v0.10.1
* Added a trialized widget for TimeSeries. [PR #232](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/232)
* Loosened upper bound version on `ipywidgets`. [PR #260](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/260)
