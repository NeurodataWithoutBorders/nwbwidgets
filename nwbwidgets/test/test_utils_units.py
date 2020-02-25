from utils.units import get_min_spike_time,get_max_spike_time

start_time = datetime(2017, 4, 3, 11, tzinfo=tzlocal())
create_date = datetime(2017, 4, 15, 12, tzinfo=tzlocal())

nwbfile = NWBFile(session_description='demonstrate NWBFile basics',  # required
                  identifier='NWB123',  # required
                  session_start_time=start_time,  # required
                  file_create_date=create_date)  # optional

nwbfile.add_unit_column('location', 'the anatomical location of this unit')
nwbfile.add_unit_column('quality', 'the quality for the inference of this unit')

nwbfile.add_unit(id=1, spike_times=[2.2, 3.0, 4.5],
                 obs_intervals=[[1, 10]], location='CA1', quality=0.95)
nwbfile.add_unit(id=2, spike_times=[2.2, 3.0, 25.0, 26.0],
                 obs_intervals=[[1, 10], [20, 30]], location='CA3', quality=0.85)
nwbfile.add_unit(id=3, spike_times=[1.2, 2.3, 3.3, 4.5],
                 obs_intervals=[[1, 10], [20, 30]], location='CA1', quality=0.90)

nwbfile.add_trial_column(name='stim', description='the visual stimuli during the trial')

nwbfile.add_trial(start_time=0.0, stop_time=2.0, stim='person')
nwbfile.add_trial(start_time=3.0, stop_time=5.0, stim='ocean')
nwbfile.add_trial(start_time=6.0, stop_time=8.0, stim='desert')



assert(get_min_spike_time(nwbfile.units)==1.2)
