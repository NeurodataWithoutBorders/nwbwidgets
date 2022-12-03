from functools import lru_cache

import numpy as np
from skimage import measure
import ipywidgets as widgets
import plotly.graph_objects as go
import plotly.express as px
from pynwb.base import NWBDataInterface
from pynwb.ophys import RoiResponseSeries, DfOverF, PlaneSegmentation, ImageSegmentation

from .base import df_to_hover_text
from .timeseries import BaseGroupedTraceWidget
from .utils.cmaps import linear_transfer_function
from .utils.dynamictable import infer_categorical_columns
from .controllers import ProgressBar


color_wheel = px.colors.qualitative.Dark24


def show_df_over_f(df_over_f: DfOverF, neurodata_vis_spec: dict):
    if len(df_over_f.roi_response_series) == 1:
        title, data_input = list(df_over_f.roi_response_series.items())[0]
        return neurodata_vis_spec[RoiResponseSeries](data_input, neurodata_vis_spec, title=title)
    else:
        return neurodata_vis_spec[NWBDataInterface](df_over_f, neurodata_vis_spec)


def show_image_segmentation(img_seg: ImageSegmentation, neurodata_vis_spec: dict):
    if len(img_seg.plane_segmentations) == 1:
        return route_plane_segmentation(list(img_seg.plane_segmentations.values())[0], neurodata_vis_spec)
    else:
        return neurodata_vis_spec[NWBDataInterface](img_seg, neurodata_vis_spec)


def show_plane_segmentation_3d_voxel(plane_seg: PlaneSegmentation):
    import ipyvolume.pylab as p3

    nrois = len(plane_seg)

    voxel_mask = plane_seg["voxel_mask"]

    mx, my, mz = 0, 0, 0
    for voxel in voxel_mask:
        for x, y, z, _ in voxel:
            mx = max(mx, x)
            my = max(my, y)
            mz = max(mz, z)

    fig = p3.figure()
    for icolor, color in enumerate(color_wheel):
        vol = np.zeros((mx + 1, my + 1, mz + 1))
        sel = np.arange(icolor, nrois, len(color_wheel))
        for isel in sel:
            dat = voxel_mask[isel]
            for x, y, z, value in dat:
                vol[x, y, z] = value
        p3.volshow(vol, tf=linear_transfer_function(color, max_opacity=0.3))
    return fig


def show_plane_segmentation_3d_mask(plane_seg: PlaneSegmentation):
    import ipyvolume.pylab as p3

    nrois = len(plane_seg)

    image_masks = plane_seg["image_mask"]

    fig = p3.figure()
    for icolor, color in enumerate(color_wheel):
        vol = np.zeros(image_masks.shape[1:])
        sel = np.arange(icolor, nrois, len(color_wheel))
        for isel in sel:
            vol += plane_seg["image_mask"][isel]
        p3.volshow(vol, tf=linear_transfer_function(color, max_opacity=0.3))
    return fig


class PlaneSegmentation2DWidget(widgets.VBox):
    def __init__(self, plane_seg: PlaneSegmentation, color_wheel=color_wheel, **kwargs):
        super().__init__()
        self.categorical_columns = infer_categorical_columns(plane_seg)
        self.plane_seg = plane_seg
        self.color_wheel = color_wheel
        self.progress_bar = ProgressBar()
        self.button = widgets.Button(description="Display ROIs")
        self.children = [widgets.HBox([self.button, self.progress_bar.container])]
        self.button.on_click(self.on_button_click)
        self.kwargs = kwargs

    def on_button_click(self, b):
        if len(self.categorical_columns) == 1:
            self.color_by = list(self.categorical_columns.keys())[0]  # changing local variables to instance variables?
            self.children += (self.show_plane_segmentation_2d(color_by=self.color_by, **self.kwargs),)
        elif len(self.categorical_columns) > 1:
            self.cat_controller = widgets.Dropdown(options=list(self.categorical_columns), description="color by")
            self.fig = self.show_plane_segmentation_2d(color_by=self.cat_controller.value, **self.kwargs)

            def on_change(change):
                if change["new"] and isinstance(change["new"], dict):
                    ind = change["new"]["index"]
                    if isinstance(ind, int):
                        color_by = change["owner"].options[ind]
                        self.update_fig(color_by)

            self.cat_controller.observe(on_change)
            self.children += (self.cat_controller, self.fig)
        else:
            self.children += (self.show_plane_segmentation_2d(color_by=None, **self.kwargs),)
        self.children = self.children[1:]

    def update_fig(self, color_by):
        cats = np.unique(self.plane_seg[color_by][:])
        legendgroups = []
        with self.fig.batch_update():
            for color_val, data in zip(self.plane_seg[color_by][:], self.fig.data):
                color = self.color_wheel[np.where(cats == color_val)[0][0]]  # store the color
                data.line.color = color  # set the color
                data.legendgroup = str(color_val)  # set the legend group to the color
                data.name = str(color_val)
            for color_val, data in zip(self.plane_seg[color_by][:], self.fig.data):
                if color_val not in legendgroups:
                    data.showlegend = True
                    legendgroups.append(color_val)
                else:
                    data.showlegend = False

    def show_plane_segmentation_2d(
        self,
        color_wheel: list = color_wheel,
        color_by: str = None,
        threshold: float = 0.01,
        fig: go.Figure = None,
        width: int = 600,
        ref_image=None,
    ):
        """
        Parameters
        ----------
        plane_seg: PlaneSegmentation
        color_wheel: list, optional
        color_by: str, optional
        threshold: float, optional
        fig: plotly.graph_objects.Figure, optional
        width: int, optional
            width of image in pixels. Height is automatically determined
            to be proportional
        ref_image: image, optional
        """
        layout_kwargs = dict()
        if color_by:
            if color_by not in self.plane_seg:
                raise ValueError("specified color_by parameter, {}, not in plane_seg object".format(color_by))
            cats = np.unique(self.plane_seg[color_by][:])
            layout_kwargs.update(title=color_by)

        data = self.plane_seg["image_mask"].data
        n_units = len(data)
        if fig is None:
            fig = go.FigureWidget()

        if ref_image is not None:
            fig.add_trace(go.Heatmap(z=ref_image, hoverinfo="skip", showscale=False, colorscale="gray"))

        aux_leg = []
        import pandas as pd

        plane_seg_hover_dict = {
            key: self.plane_seg[key].data for key in self.plane_seg.colnames if key not in ["pixel_mask", "image_mask"]
        }
        plane_seg_hover_dict.update(id=self.plane_seg.id.data)
        plane_seg_hover_df = pd.DataFrame(plane_seg_hover_dict)
        all_hover = df_to_hover_text(plane_seg_hover_df)
        self.progress_bar.reset(total=n_units)
        self.progress_bar.set_description("Loading Image Masks")
        for i in range(n_units):
            kwargs = dict(showlegend=False)
            if color_by is not None:
                if plane_seg_hover_df[color_by][i] not in aux_leg:
                    kwargs.update(showlegend=True)
                    aux_leg.append(plane_seg_hover_df[color_by][i])
                index = np.where(cats == plane_seg_hover_df[color_by][i])[0][0]
                c = color_wheel[index % len(color_wheel)]
                kwargs.update(
                    line_color=c,
                    name=str(plane_seg_hover_df[color_by][i]),
                    legendgroup=str(plane_seg_hover_df[color_by][i]),
                )

            x, y = self.compute_outline(i, threshold)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    fill="toself",
                    mode="lines",
                    text=all_hover[i],
                    hovertext="text",
                    line=dict(width=0.5),
                    **kwargs,
                )
            )
            self.progress_bar.update()
        # self.progress_bar.close()
        fig.update_layout(
            width=width,
            yaxis=dict(
                mirror=True,
                scaleanchor="x",
                scaleratio=1,
                range=[0, self.plane_seg["image_mask"].shape[2]],
                constrain="domain",
            ),
            xaxis=dict(
                mirror=True,
                range=[0, self.plane_seg["image_mask"].shape[1]],
                constrain="domain",
            ),
            margin=dict(t=30, b=10),
            **layout_kwargs,
        )
        return fig

    @lru_cache(1000)
    def compute_outline(self, i, threshold):
        x, y = zip(*measure.find_contours(self.plane_seg["image_mask"][i], threshold)[0])
        return x, y


def route_plane_segmentation(plane_seg: PlaneSegmentation, neurodata_vis_spec: dict):
    if "voxel_mask" in plane_seg:
        return show_plane_segmentation_3d_voxel(plane_seg)
    elif "image_mask" in plane_seg and len(plane_seg.image_mask.shape) == 4:
        raise NotImplementedError("3d image mask vis not implemented yet")
    elif "image_mask" in plane_seg:
        return PlaneSegmentation2DWidget(plane_seg)


class RoiResponseSeriesWidget(BaseGroupedTraceWidget):
    def __init__(self, roi_response_series: RoiResponseSeries, neurodata_vis_spec=None, **kwargs):
        super().__init__(roi_response_series, "rois", **kwargs)
