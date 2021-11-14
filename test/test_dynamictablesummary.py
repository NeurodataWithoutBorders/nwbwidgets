import unittest
from datetime import datetime
from dateutil.tz import tzlocal

from ipywidgets import widgets
from pynwb import NWBFile
from nwbwidgets.dynamictablesummary import DynamicTableSummaryWidget


class DynamicSummaryTestCase(unittest.TestCase):
    def setUp(self):
        """
        Test with unit tables
        """
        start_time = datetime(2017, 4, 3, 11, tzinfo=tzlocal())
        create_date = datetime(2017, 4, 15, 12, tzinfo=tzlocal())

        self.nwbfile = NWBFile(
            session_description="NWBFile for DynamicTableSummaryWidget",
            identifier="NWB123",
            session_start_time=start_time,
            file_create_date=create_date,
        )

        self.nwbfile.add_unit_column(
            "location", "the anatomical location of this unit")
        self.nwbfile.add_unit_column(
            "quality", "the quality for the inference of this unit"
        )
        self.nwbfile.add_unit_column(
            "isi_violation", "the ISI violation ratio of this unit"
        )
        self.nwbfile.add_unit_column(
            "firing_rate", "the firing rate of this unit"
        )

        self.nwbfile.add_unit(
            spike_times=[2.2, 3.0, 4.5],
            obs_intervals=[[1, 10]],
            location="CA1",
            quality=0.95,
            isi_violation=0.1,
            firing_rate=5.1
        )
        self.nwbfile.add_unit(
            spike_times=[2.2, 3.0, 25.0, 26.0],
            obs_intervals=[[1, 10], [20, 30]],
            location="CA3",
            quality=0.85,
            isi_violation=0.12,
            firing_rate=2.1
        )
        self.nwbfile.add_unit(
            spike_times=[1.2, 2.3, 3.3, 4.5],
            obs_intervals=[[1, 10], [20, 30]],
            location="CA1",
            quality=0.90,
            isi_violation=0.3,
            firing_rate=10.1
        )
        self.nwbfile.add_unit(
            spike_times=[1.4, 2.4, 3.4, 4.6],
            obs_intervals=[[1, 10], [20, 30]],
            location="CA3",
            quality=0.92,
            isi_violation=0.21,
            firing_rate=14.4
        )

    def test_psth_widget(self):
        widget = DynamicTableSummaryWidget(self.nwbfile.units)
        assert isinstance(widget, widgets.Widget)

        nbins = 2
        col_names_display = widget.col_names_display.keys()
        real_names = [col for col in col_names_display if "(r)" in col]
        cat_names = [col for col in col_names_display if "(c)" in col]
        
        # select 1 real
        cols_to_plot = (real_names[0])
        widget.controls["col_names_display"] = cols_to_plot
        widget.plot_hist_bar(col_names_display=cols_to_plot, nbins=nbins)
        
        # select 1 cat
        cols_to_plot = (cat_names[0])
        widget.controls["col_names_display"] = cols_to_plot
        widget.plot_hist_bar(col_names_display=cols_to_plot, nbins=nbins)
        
        # select 1 real + 1 cat
        cols_to_plot = (real_names[0], cat_names[0])
        widget.controls["col_names_display"] = cols_to_plot
        widget.plot_hist_bar(col_names_display=cols_to_plot, nbins=nbins)
        
        # select 2 real
        cols_to_plot = (real_names[0], real_names[1])
        widget.controls["col_names_display"] = cols_to_plot
        widget.plot_hist_bar(col_names_display=cols_to_plot, nbins=nbins)
        
        # select 2 real + 1 cat
        cols_to_plot = (real_names[0], real_names[1], cat_names[0])
        widget.controls["col_names_display"] = cols_to_plot
        widget.plot_hist_bar(col_names_display=cols_to_plot, nbins=nbins)
        
        # select 3 real
        cols_to_plot = (real_names[0], real_names[1], real_names[2])
        widget.controls["col_names_display"] = cols_to_plot
        widget.plot_hist_bar(col_names_display=cols_to_plot, nbins=nbins)
        

        # select 3 real + 1 cat
        cols_to_plot = (real_names[0], real_names[1],
                        real_names[2], cat_names[0])
        widget.controls["col_names_display"] = cols_to_plot
        widget.plot_hist_bar(col_names_display=cols_to_plot, nbins=nbins)
