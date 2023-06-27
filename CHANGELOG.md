# Upcoming

### New Features
* Improvements on Panel Docker file, including path for read-only mount to access local files [PR #299](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/299)

### Fixes
* Fix I/O issues when streaming data on Panel [PR #295](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/295)
* Fix plotly Figure not showing up, pinned Plotly version [PR #297](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/297)
* Fix BehavioralTimeSeries not showing up [PR #297](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/297)


# v0.10.2

### Fixes
* Prevented the display of video assets on DANDI from the `Panel` dropdown. [PR #281](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/281)
* Remove `trials` from the accordion of `nwb2widget` (it will display in the `intervals` tab alongside any other `TimeIntervals`). [PR #281](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/281)
* Prevent the `ElectrodeGroupWidget` from loading if positions (specifically, `x`) are missing in conjunction with nwb-schema versions that allow those columns to be optional. [PR #280](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/280)



# v0.10.1

### New Features
* Added a trialized widget for TimeSeries. [PR #232](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/232)

### Dependencies
* Loosened upper bound version on `ipywidgets`. [PR #260](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/260)
