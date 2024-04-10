import numpy as np

from nwbwidgets.analysis.spikes import compute_smoothed_firing_rate


def test_compute_smoothed_firing_rate():
    spike_times = np.array([1.0, 2.0, 5.0, 5.5, 7.0, 7.5, 8.0])
    tt = np.arange(10, dtype=float)
    expected_binned_spikes = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0])
    expected_smoothed_spikes = np.array(
        [
            0.35438556,
            0.64574827,
            0.64991247,
            0.40421249,
            0.55136343,
            0.91486675,
            1.02201074,
            1.14797447,
            0.89644961,
            0.41307621,
        ]
    )
    binned_spikes = compute_smoothed_firing_rate(spike_times, tt, 0.001)
    smoothed_spikes = compute_smoothed_firing_rate(spike_times, tt, 1.0)
    np.testing.assert_allclose(expected_binned_spikes, binned_spikes)
    np.testing.assert_allclose(expected_smoothed_spikes, smoothed_spikes)
