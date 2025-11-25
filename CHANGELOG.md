# Upcoming

# v0.12.0

### New Features
* Enable / disable caching option on Panel [PR #316](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/316)


# v0.11.0

### New Features
* Improvements on Panel Docker file, including path for read-only mount to access local files [PR #299](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/299)
* Panel error handling with message output [PR #299](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/299)
* New flag argument on Panel to enable/disable warnings [PR #299](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/299)
* Improved browsing and reading local NWB files with Panel, using ipyfilechooser [PR #300](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/300)
* Improve readibility of dandisets and files dropdown lists [PR #301](https://github.com/NeurodataWithoutBorders/nwbwidgets/pull/301)

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
