import numpy as np
import matplotlib.pyplot as plt
from pynwb.ophys import RoiResponseSeries, DfOverF, PlaneSegmentation, TwoPhotonSeries, ImageSegmentation
from pynwb.base import NWBDataInterface
from ndx_grayscalevolume import GrayscaleVolume
from collections import OrderedDict
from .utils.cmaps import linear_transfer_function
import ipywidgets as widgets
from itertools import cycle
from matplotlib import colors
from .base import show_neurodata_base


color_wheel = ['red', 'green', 'black', 'blue', 'magenta', 'yellow']


def show_two_photon_series(indexed_timeseries: TwoPhotonSeries, neurodata_vis_spec: OrderedDict):
    output = widgets.Output()

    if len(indexed_timeseries.data.shape) == 3:
        def show_image(index=0):
            fig, ax = plt.subplots(subplot_kw={'xticks': [], 'yticks': []})
            ax.imshow(indexed_timeseries.data[index], cmap='gray')
            output.clear_output(wait=True)
            with output:
                plt.show(fig)
    elif len(indexed_timeseries.data.shape) == 4:
        import ipyvolume.pylab as p3

        def show_image(index=0):
            fig = p3.figure()
            p3.volshow(indexed_timeseries.data[index], tf=linear_transfer_function([0, 0, 0], max_opacity=.3))
            output.clear_output(wait=True)
            with output:
                p3.show()
    else:
        raise NotImplementedError

    def on_index_change(change):
        show_image(change.new)

    slider = widgets.IntSlider(value=0, min=0,
                               max=indexed_timeseries.data.shape[0] - 1,
                               orientation='horizontal')
    slider.observe(on_index_change, names='value')
    show_image()

    return widgets.VBox([output, slider])


def show_df_over_f(df_over_f: DfOverF, neurodata_vis_spec: OrderedDict):
    if len(df_over_f.roi_response_series) == 1:
        title, input = list(df_over_f.roi_response_series.items())[0]
        return neurodata_vis_spec[RoiResponseSeries](input, neurodata_vis_spec, title=title)
    else:
        return neurodata_vis_spec[NWBDataInterface](df_over_f, neurodata_vis_spec)


def show_roi_response_series(roi_response_series: RoiResponseSeries, neurodata_vis_spec: OrderedDict,
                             nchans: int = 30, title: str = None):
    """

    :param roi_response_series: pynwb.ophys.RoiResponseSeries
    :param neurodata_vis_spec: OrderedDict
    :param nchans: int
    :param title: str
    :return: matplotlib.pyplot.Figure
    """
    tt = roi_response_series.timestamps
    data = roi_response_series.data
    if data.shape[1] == len(tt):  # fix of orientation is incorrect
        mini_data = data[:nchans, :].T
    else:
        mini_data = data[:, :nchans]

    gap = np.median(np.nanstd(mini_data, axis=0)) * 20
    offsets = np.arange(nchans) * gap

    fig, ax = plt.subplots()
    ax.plot(tt, mini_data + offsets)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('traces (first 30)')
    if np.isfinite(gap):
        ax.set_ylim(-gap, offsets[-1] + gap)
        ax.set_xlim(tt[0], tt[-1])
        ax.set_yticks(offsets)
        ax.set_yticklabels(np.arange(mini_data.shape[1]))

    if title is not None:
        ax.set_title(title)

    return fig


def show_image_segmentation(img_seg: ImageSegmentation, neurodata_vis_spec: OrderedDict):
    if len(img_seg.plane_segmentations) == 1:
        return show_plane_segmentation(next(iter(img_seg.plane_segmentations.values())), neurodata_vis_spec)
    else:
        return show_neurodata_base(ImageSegmentation, neurodata_vis_spec)


def show_plane_segmentation(plane_seg: PlaneSegmentation, neurodata_vis_spec: OrderedDict):
    nrois = len(plane_seg)

    if 'voxel_mask' in plane_seg:
        import ipyvolume.pylab as p3

        dims = np.array([max(max(plane_seg['voxel_mask'][i][dim]) for i in range(nrois))
                         for dim in ['x', 'y', 'z']]).astype('int') + 1
        fig = p3.figure()
        for icolor, color in enumerate(color_wheel):
            vol = np.zeros(dims)
            sel = np.arange(icolor, nrois, len(color_wheel))
            for isel in sel:
                dat = plane_seg['voxel_mask'][isel]
                vol[tuple(dat['x'].astype('int')),
                    tuple(dat['y'].astype('int')),
                    tuple(dat['z'].astype('int'))] = 1
            p3.volshow(vol, tf=linear_transfer_function(color, max_opacity=.3))
        return fig
    elif 'image_mask' in plane_seg:
        data = plane_seg['image_mask'].data
        img = np.ones(shape=list(data.shape[1:]) + [3])
        for c, img_mask in zip(cycle(color_wheel), data):
            img[img_mask.astype(bool), :] = colors.to_rgb(c)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

        return fig


def show_grayscale_volume(vol: GrayscaleVolume, neurodata_vis_spec: OrderedDict):
    import ipyvolume.pylab as p3

    fig = p3.figure()
    p3.volshow(vol.data, tf=linear_transfer_function([0, 0, 0], max_opacity=.1))
    return fig
