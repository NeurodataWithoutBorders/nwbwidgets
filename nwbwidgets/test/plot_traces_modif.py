
def plot_traces(time_series: TimeSeries, time_start=0, time_duration=None, trace_window=None,
                title: str = None, ylabel: str = 'traces'):
    """
    Parameters
    ----------
    time_series: pynwb.TimeSeries
    time_start: float
        Start time in seconds
    time_duration: float, optional
        Duration in seconds. Default:
    trace_window: [int int], optional
        Index range of traces to view
    title: str, optional
    ylabel: str, optional
    Returns
    -------
    """
    
    if type(time_series.data) != np.ndarray:
        time_series.data = np.asarray(time_series.data)
    
    if time_start == 0:
        t_ind_start = 0
    else:
        t_ind_start = timeseries_time_to_ind(time_series, time_start)
    
    if time_duration is None:
        t_ind_stop = None
    else:
        t_ind_stop = timeseries_time_to_ind(time_series, time_start + time_duration)
    
    if t_ind_stop == None:
        tt = get_timeseries_tt(time_series, t_ind_start, time_series.data.shape[-1])
    else:
        tt = get_timeseries_tt(time_series, t_ind_start, t_ind_stop)
    
    if trace_window is None:
        trace_window = [0, time_series.data.shape[-1]]
    
    mini_data = time_series.data[:, t_ind_start:t_ind_stop]
    
    gap = np.median(np.nanstd(mini_data, axis=0)) * 20
    offsets = np.arange(t_ind_stop - t_ind_start) * gap
    fig, ax = plt.subplots()
    ax.figure.set_size_inches(12, 6)
    ax.plot(tt, (mini_data + offsets).T)
    ax.set_xlabel('time (s)')
    if np.isfinite(gap):
        ax.set_ylim(-gap, offsets[-1] + gap)
        ax.set_xlim(tt[0], tt[-1])
        ax.set_yticks(offsets)
        ax.set_yticklabels(np.arange(trace_window[0], trace_window[1]))
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return fig
