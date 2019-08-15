import numpy as np
import matplotlib.pyplot as plt
from pynwb.ophys import RoiResponseSeries
from pynwb.base import NWBDataInterface


def show_df_over_f(df_over_f, neurodata_vis_spec):
    if len(df_over_f.roi_response_series) == 1:
        title, input = list(df_over_f.roi_response_series.items())[0]
        return neurodata_vis_spec[RoiResponseSeries](input, neurodata_vis_spec, title=title)
    else:
        return neurodata_vis_spec[NWBDataInterface](df_over_f, neurodata_vis_spec)


def show_roi_response_series(roi_response_series, neurodata_vis_spec, nchans=30, title=None):
    """

    :param roi_response_series:
    :param neurodata_vis_spec:
    :param nchans:
    :param title:
    :return:
    """
    tt = roi_response_series.timestamps
    mini_data = roi_response_series.data[:, :nchans]

    gap = np.median(np.std(mini_data, axis=0)) * 10
    offsets = np.arange(nchans) * gap

    fig, ax = plt.subplots()
    ax.plot(tt, mini_data + offsets)
    ax.set_ylim(-gap, offsets[-1] + gap)
    ax.set_xlim(tt[0], tt[-1])
    ax.set_yticks(offsets)
    ax.set_yticklabels(np.arange(mini_data.shape[1]))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('traces (first 30)')

    if title is not None:
        ax.set_title(title)

    return fig
