import ipywidgets as widgets
import pynwb
from .view import default_neurodata_vis_spec
import spikeextractors as se
from pynwb.ecephys import LFP


def _set_spec():
    ephys_viz_neurodata_vis_spec = dict(default_neurodata_vis_spec)
    ephys_viz_neurodata_vis_spec[pynwb.ecephys.LFP] = show_lfp


def show_lfp(node: LFP, **kwargs):
    import spikeextractors as se
    import ephys_viz as ev
    try:
        recording = LFPRecordingExtractor(lfp_node=node)
    except:
        return widgets.Text('Problem creating LFPRecordingExtractor')
    return ev.TimeseriesView(
        recording=recording,
        initial_y_scale_factor=5
    ).show(render=False)


class LFPRecordingExtractor(se.RecordingExtractor):
    def __init__(self, lfp_node: LFP):
        super().__init__()
        lfp = list(lfp_node.electrical_series.values())[0]    
        self._samplerate = lfp.rate
        self._data = lfp.data
        self._num_channels = self._data.shape[1]
        self._num_timepoints = self._data.shape[0]

    def get_channel_ids(self):
        return list(range(self._num_channels))

    def get_num_frames(self):
        return self._num_timepoints

    def get_sampling_frequency(self):
        return self._samplerate

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        return self._data[start_frame:end_frame, :][:, channel_ids].T

_set_spec()