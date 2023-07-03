import plotly.graph_objects as go
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from hdmf.common import DynamicTable, VectorData
from pynwb import NWBFile, TimeSeries


def get_partial_path_from_parents(obj):
    # ndx-events do not have object.data
    parents = []
    while obj is not None and obj.name != "root":
        parents.append(obj.name)
        obj = obj.parent  # assuming the object has a 'parent' attribute
    return parents


def extract_timing_information_from_nwbfile(nwbfile: NWBFile, verbose: bool = False) -> dict:
    temporal_information_dict = {}
    for object_id, object in nwbfile.objects.items():
        object_name = object.name
        if hasattr(object, "data") and hasattr(object.data, "name"):
            object_path = object.data.name
            top_module = object_path[1:].split("/")[0]
            if top_module == "processing":
                top_module = object_path[1:].split("/")[1]

        elif hasattr(object, "data"):
            object_path = get_partial_path_from_parents(object)[::-1]
            top_module = object_path[0] if object_path else ""
        else:
            object_path = get_partial_path_from_parents(object)[::-1]
            top_module = object_path[0] if object_path else ""

        if isinstance(object, DynamicTable) and hasattr(object, "start_time") and hasattr(object, "stop_time"):
            if verbose:
                print("------------------")
                print(f"branch: dynamic table with intervals for object {object_name} with path {object_path}")
                print(f"Top module {top_module}")
                print("------------------")

            start_time = object.start_time.data[:]
            stop_time = object.stop_time.data[:]

            intervals = [
                {"start_time": start_time, "stop_time": stop_time}
                for start_time, stop_time in zip(start_time, stop_time)
            ]
            timing_dict = dict(start_time=intervals[0]["start_time"], stop_time=intervals[-1]["stop_time"])
            object_info = dict(
                object_path=object_path, object_name=object_name, object_id=object_id, top_module=top_module
            )
            temporal_information_dict[object_name] = dict(
                timing_dict=timing_dict, info=object_info, intervals=intervals
            )

        elif isinstance(object, DynamicTable) and hasattr(object, "start_time"):
            if verbose:
                print("------------------")
                print("NOT DONE YET")
                print(f"Branch of dynamic table with only start_time for object {object_name} with path {object_path}")
                print(f"Top module {top_module}")
                print("------------------")

        elif isinstance(object, TimeSeries):
            time_series = object
            if verbose:
                print("------------------")
                print(f"branch: time series for object {object_name} with path {object_path}")
                print(f"Top module {top_module}")
                print("------------------")
            timing_dict = {}
            if time_series.timestamps is not None:
                timing_dict["start_time"] = time_series.timestamps[0]
                timing_dict["stop_time"] = time_series.timestamps[-1]
            else:
                timing_dict["start_time"] = time_series.starting_time
                timing_dict["stop_time"] = time_series.starting_time + time_series.num_samples / time_series.rate

            # TODO This fails when the series has timestamps with gaps. One option to find out the gaps and write the series
            # As intervals is that the naive way np.diff generates a too large memory allocation.
            # Probably, a bisection algorithm can be used to find the gaps.

            object_info = dict(
                object_path=object_path, object_name=object_name, object_id=object_id, top_module=top_module
            )
            temporal_information_dict[object_name] = dict(timing_dict=timing_dict, info=object_info)

    return temporal_information_dict


def generate_epoch_guides_trace(temporal_data_dict):
    epoch_dict = temporal_data_dict["epochs"]
    intervals = epoch_dict["intervals"]
    x_epoch = []
    y_epoch = []
    for i, epoch in enumerate(intervals, start=1):
        start_time = epoch["start_time"]
        stop_time = epoch["stop_time"]
        x_epoch.extend([start_time, start_time, None])
        x_epoch.extend([stop_time, stop_time, None])
        y_epoch.extend([0.9, len(temporal_data_dict) + 1, None])
        y_epoch.extend([0.9, len(temporal_data_dict) + 1, None])

    trace = go.Scatter(
        x=x_epoch,
        y=y_epoch,
        mode="lines",
        line=dict(color="#808080", width=1.5, dash="dash"),
        name="epochs_guides",
        visible=False,
    )

    return trace


def generate_epoch_annotations(temporal_data_dict) -> list:
    if "epochs" not in temporal_data_dict:
        return []

    epoch_dict = temporal_data_dict["epochs"]
    intervals = epoch_dict["intervals"]

    annotation_dict_list = []
    for i, epoch in enumerate(intervals, start=1):
        start_time = epoch["start_time"]
        stop_time = epoch["stop_time"]

        # Add a text box for each epoch
        annotation_x = start_time + (stop_time - start_time) / 2
        annotation_y = len(temporal_data_dict) + len(temporal_data_dict) * 0.05
        annotation_dict = dict(
            x=annotation_x,
            y=annotation_y,
            text=f"Epoch {i}",
            showarrow=False,
            font=dict(size=8, color="#000000"),
            align="center",
            borderwidth=2,
            borderpad=4,
            opacity=0.6,
        )
        annotation_dict_list.append(annotation_dict)
    return annotation_dict_list


def generate_total_duration_traces(temporal_data_dict, styling_dict):
    traces_dict = dict()
    for i, object_name in enumerate(styling_dict, start=1):
        times_info_dict = temporal_data_dict[object_name]
        info_dict = times_info_dict["info"]
        timing_dict = times_info_dict["timing_dict"]
        start_time = timing_dict["start_time"]
        stop_time = timing_dict["stop_time"]
        top_module = info_dict["top_module"]
        color = styling_dict[object_name]["color"]

        trace = go.Scatter(
            x=[start_time, stop_time],
            y=[i, i],
            mode="lines",
            line=dict(color=color, width=6),
            name=f"{top_module} - {object_name}",
            legendgroup=top_module,  # Group by top_module
            hovertemplate=f"{object_name}<br>start: {start_time}<br>end: {stop_time}<extra></extra>",
        )
        traces_dict[object_name] = trace

    return traces_dict


def generate_time_grid_widget(temporal_data_dict: dict) -> go.FigureWidget:
    fig = go.FigureWidget()

    # Key modules for temporal organization ad trials and epochs
    key_modules = ["epochs", "trials"]
    sorted_keys = [key for key in temporal_data_dict.keys() if key not in key_modules]
    sorted_keys.sort(key=lambda x: temporal_data_dict[x]["info"]["top_module"])
    unique_top_modules = list(set([temporal_data_dict[key]["info"]["top_module"] for key in sorted_keys]))

    # if "epochs" in temporal_data_dict:
    #     sorted_keys = ["epochs"] + sorted_keys
    # if "trials" in temporal_data_dict:
    #     sorted_keys = sorted_keys + ["trials"]

    # Generate a color map for the top modules
    num_top_modules = len(unique_top_modules)
    cm = plt.get_cmap("Accent")  # Use 'gist_rainbow' colormap
    cNorm = colors.Normalize(vmin=0, vmax=num_top_modules - 1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    color_map = {module: colors.rgb2hex(scalarMap.to_rgba(i)) for i, module in enumerate(unique_top_modules)}

    styling_dict_epochs = dict(epochs=dict(color="#d3d3d3"))
    styling_dict_rest = {
        key: dict(color=color_map[temporal_data_dict[key]["info"]["top_module"]]) for key in sorted_keys
    }
    styling_dict_trials = dict(trials=dict(color="#d3d3d3"))

    styling_dict = dict()
    if "trials" in temporal_data_dict:
        styling_dict.update(styling_dict_trials)

    styling_dict.update(styling_dict_rest)

    if "epochs" in temporal_data_dict:
        styling_dict.update(styling_dict_epochs)

    traces_dict = generate_total_duration_traces(temporal_data_dict, styling_dict)
    for trace in traces_dict.values():
        fig.add_trace(trace)

    epoch_guide_trace = generate_epoch_guides_trace(temporal_data_dict)
    fig.add_trace(epoch_guide_trace)

    annotation_dict_list = generate_epoch_annotations(temporal_data_dict)
    # Button for toggling epoch guides
    epoch_guide_button = dict(
        type="buttons",
        direction="down",
        showactive=True,
        buttons=[
            dict(
                label="Toggle Epoch Guides",
                method="update",
                args=[{"visible": [True] * len(traces_dict) + [False]}, {"annotations": []}],
                args2=[{"visible": [True] * len(traces_dict) + [True]}, {"annotations": annotation_dict_list}],
            ),
        ],
        x=0.10,  # Position from left (0 to 1)
        y=-0.15,  # Position from bottom (0 to 1)
    )

    ticktext = list(styling_dict.keys())
    tickvals = list(range(1, len(temporal_data_dict) + 1))
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext,
            range=[0.5, len(ticktext) * 1.25],
            ticks="",
            showline=False,
            showticklabels=True,
            showgrid=True,
            gridwidth=0.1,
        ),
        updatemenus=[epoch_guide_button],
    )

    return fig


from ipywidgets import Layout, fixed, widgets


class TimeGrid(widgets.VBox):
    def __init__(self, nwbfile: NWBFile):
        super().__init__()

        self.nwbfile = nwbfile
        # This is the calculation part
        self.temporal_data_dict = extract_timing_information_from_nwbfile(nwbfile=self.nwbfile)

        # This is the figure widget
        self.figure_widget = generate_time_grid_widget(temporal_data_dict=self.temporal_data_dict)
        self.children = [self.figure_widget]
