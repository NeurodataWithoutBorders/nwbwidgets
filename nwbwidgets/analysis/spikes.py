import numpy as np
import scipy


def compute_smoothed_firing_rate(spike_times, tt, sigma_in_secs):
    """Evaluate gaussian smoothing of spike_times at uniformly spaced array t
    Args:
      spike_times:
          A 1D numpy ndarray of spike times
      tt:
          1D array, uniformly spaced, e.g. the output of np.linspace or np.arange
      sigma_in_secs:
          standard deviation of the smoothing gaussian in seconds
    Returns:
          Gaussian smoothing evaluated at array t
    """
    if len(spike_times) < 2:
        return np.zeros_like(tt)
    binned_spikes = np.zeros_like(tt)
    binned_spikes[np.searchsorted(tt, spike_times)] += 1
    dt = np.diff(tt[:2])[0]
    sigma_in_samps = sigma_in_secs / dt
    smooth_fr = scipy.ndimage.gaussian_filter1d(binned_spikes, sigma_in_samps) / dt
    return smooth_fr


# ported from the chronux MATLAB package
def psth(data=None, sig=0.05, T=None, err=2, t=None, num_bootstraps=1000):
    """Find peristimulus time histogram smoothed by a gaussian kernel
        The time units of the arrays in data, sig and t
        should be the same, e.g. seconds
    Args:
      data:
            A dictionary of channel names, and 1D numpy ndarray spike times,
            A numpy ndarray, where each row gives the spike times for each channel
            A 1D numpy ndarray, one list or tuple of floats that gives spike times for only one channel
      sig:  standard deviation of the smoothing gaussian. default 0.05
      T: time interval [a,b], spike times strictly outside this interval are excluded
      err: An integer, 0, 1, or 2. default 2
            0 indicates no standard error computation
            1 Poisson error
            2 Bootstrap method over trials
      t: 1D array, list or tuple indicating times to evaluate psth at
      num_bootstraps: number of bootstraps. Effective only in computing error when err=2. default 10
    Returns:
      R: Rate, mean smoothed peristimulus time histogram
      t: 1D array, list or tuple indicating times psth is evaluated at
      E: standard error
    """

    # verify data argument
    try:
        if isinstance(data, dict):
            data = data.values()
        elif isinstance(data[0], (float, int)):
            data = np.array([data])
        elif isinstance(data[0], (np.ndarray, list, tuple)):
            data = np.array(data)
        else:
            raise TypeError
        if isinstance(data, np.ndarray) and len(data.shape) == 2:
            if data.shape[0] == 0 or data.shape[1] == 0:
                raise TypeError
        else:
            data = [np.array(ch_data) for ch_data in data]
            for ch_data in data:
                if len(ch_data.shape) != 1 or not isinstance(ch_data[0], (float, int)):
                    raise TypeError
    except Exception as exc:
        msg = (
            "psth requires spike time data as first positional argument. "
            + "Spike time data should be in the form of:\n"
            + "   a dictionary of channel names, and 1D numpy ndarray spike times\n"
            + "   a 2D numpy ndarray, where each row represents the spike times of a channel",
        )
        exc.args = msg
        raise exc

    # input data size
    num_t = len(data)  # number of trials
    channel_lengths = [len(ch_data) for ch_data in data]

    if not isinstance(sig, (float, int)) or sig <= 0:
        raise TypeError(
            "sig must be positive. Only the non-adaptive method is supported"
        )
    if not isinstance(num_bootstraps, int) or num_bootstraps <= 0:
        raise TypeError("num_bootstraps must be a positive integer")

    # determine the interval of interest T, and mask times outside of the interval
    if T is not None:
        # expand T to avoid edge effects in rate
        T = [T[0] - 4 * sig, T[1] + 4 * sig]
        data = [np.ma.masked_outside(np.ravel(ch_data), T[0], T[1]) for ch_data in data]
    else:
        T = [
            np.ma.min([np.ma.min(c) for c in data]),
            np.ma.max([np.ma.max(c) for c in data]),
        ]

    # determine t
    if t is None:
        num_points = int(5 * (T[1] - T[0]) / sig)
        t = np.linspace(T[0], T[1], num_points)
    else:
        t = np.ravel(t)
        num_points = len(t)

    # masked input data size
    data_lengths = [np.ma.count(ch_data) for ch_data in data]
    num_times_total = sum(data_lengths) + 1

    # warn if spikes have low density
    L = num_times_total / (num_t * (T[1] - T[0]))
    if 2 * L * num_t * sig < 1 or L < 0.1:
        print(
            "Spikes have very low density. The time units may not be the same, or the kernel width is too small"
        )
        print(
            "Total events: %f \nsig: %f ms \nT: %f \nevents*sig: %f\n"
            % (num_times_total, sig * 1000, T, num_times_total * sig / (T[1] - T[0]))
        )

    # evaluate kernel density estimation at array t
    RR = np.zeros((num_t, num_points))
    for n in range(num_t):
        spike_times = data[n] if not np.ma.is_masked(data[n]) else data[n].compressed()
        RR[n, :] = compute_smoothed_firing_rate(spike_times, t, sig)

    # find rate
    R = np.mean(RR, axis=0)

    # find error
    if num_t < 4 and err == 2:
        print(
            "Switching to Poisson errorbars as number of trials is too small for bootstrap"
        )
        err = 1
    # std dev is sqrt(rate*(integral over kernal^2)/trials)
    # for Gaussian integral over Kernal^2 is 1/(2*sig*srqt(pi))
    if err == 0:
        E = None
    elif err == 1:
        E = np.sqrt(R / (2 * num_t * sig * np.sqrt(np.pi)))
    elif err == 2:
        mean_ = [
            np.mean(RR[np.random.randint(0, num_t), :]) for _ in range(num_bootstraps)
        ]
        E = np.std(mean_)
    else:
        raise TypeError("err must be 0, 1, or 2")

    return R, t, E
