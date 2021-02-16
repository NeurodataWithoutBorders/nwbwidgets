import unittest
from datetime import datetime

import ipywidgets as widgets
import numpy as np
from dateutil.tz import tzlocal
from ndx_grayscalevolume import GrayscaleVolume
from nwbwidgets.ophys import show_grayscale_volume, TwoPhotonSeriesWidget, show_df_over_f
from nwbwidgets.view import default_neurodata_vis_spec
from pynwb import NWBFile
from pynwb.device import Device
from pynwb.ophys import TwoPhotonSeries, OpticalChannel, ImageSegmentation, Fluorescence, DfOverF
from nwbwidgets.ophys import show_grayscale_volume,show_two_photon_series,show_df_over_f,show_plane_segmentation_2d,\
plane_segmentation_2d_widget,show_plane_segmentation_3d,show_plane_segmentation,show_image_segmentation
import unittest


def test_show_grayscale_volume():
    vol = GrayscaleVolume(name='vol', data=np.random.rand(2700).reshape((30, 30, 3)))
    assert isinstance(show_grayscale_volume(vol, default_neurodata_vis_spec), widgets.Widget)


class CalciumImagingTestCase(unittest.TestCase):

    def setUp(self):
        nwbfile = NWBFile('my first synthetic recording', 'EXAMPLE_ID', datetime.now(tzlocal()),
                          experimenter='Dr. Bilbo Baggins',
                          lab='Bag End Laboratory',
                          institution='University of Middle Earth at the Shire',
                          experiment_description=('I went on an adventure with thirteen '
                                                  'dwarves to reclaim vast treasures.'),
                          session_id='LONELYMTN')

        device = Device('imaging_device_1')
        nwbfile.add_device(device)
        optical_channel = OpticalChannel('my_optchan', 'description', 500.)
        imaging_plane = nwbfile.create_imaging_plane(
            name='imgpln1',
            optical_channel=optical_channel,
            description='a fake ImagingPlane',
            device=device,
            excitation_lambda=600.,
            imaging_rate=300.,
            indicator='GFP',
            location='somewhere in the brain',
            reference_frame='unknown',
            origin_coords=[10, 20],
            origin_coords_unit='millimeters',
            grid_spacing=[0.001, 0.001],
            grid_spacing_unit='millimeters')

        self.image_series = TwoPhotonSeries(name='test_iS', dimension=[2], data=np.random.rand(10, 5, 5, 3),
                                            external_file=['images.tiff'], imaging_plane=imaging_plane,
                                            starting_frame=[0], format='tiff', starting_time=0.0, rate=1.0)
        nwbfile.add_acquisition(self.image_series)

        mod = nwbfile.create_processing_module('ophys', 'contains optical physiology processed data')
        self.img_seg = ImageSegmentation()
        mod.add(self.img_seg)
        self.ps = self.img_seg.create_plane_segmentation('output from segmenting my favorite imaging plane',
                                               imaging_plane, 'my_planeseg', self.image_series)

        w, h = 3, 3
        pix_mask1 = [(0, 0, 1.1), (1, 1, 1.2), (2, 2, 1.3)]
        vox_mask1 = [(0, 0, 0, 1.1), (1, 1, 1, 1.2), (2, 2, 2, 1.3)]
        img_mask1 = [[0.0 for _ in range(w)] for _ in range(h)]
        img_mask1[0][0] = 1.1
        img_mask1[1][1] = 1.2
        img_mask1[2][2] = 1.3
        self.ps.add_roi(pixel_mask=pix_mask1, image_mask=img_mask1, voxel_mask=vox_mask1)

        pix_mask2 = [(0, 0, 2.1), (1, 1, 2.2)]
        vox_mask2 = [(0, 0, 0, 2.1), (1, 1, 1, 2.2)]
        img_mask2 = [[0.0 for _ in range(w)] for _ in range(h)]
        img_mask2[0][0] = 2.1
        img_mask2[1][1] = 2.2
        self.ps.add_roi(pixel_mask=pix_mask2, image_mask=img_mask2, voxel_mask=vox_mask2)

        fl = Fluorescence()
        mod.add(fl)

        rt_region = self.ps.create_roi_table_region('the first of two ROIs', region=[0])

        data = np.random.randn(10, 5)
        timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        rrs = fl.create_roi_response_series('my_rrs', data, rt_region, unit='lumens', timestamps=timestamps)

        self.df_over_f = DfOverF(rrs)

    def test_show_two_photon_series(self):
        assert isinstance(TwoPhotonSeriesWidget(self.image_series, default_neurodata_vis_spec), widgets.Widget)

    def test_show_df_over_f(self):
        assert isinstance(show_df_over_f(self.df_over_f, default_neurodata_vis_spec), widgets.Widget)
        
    def test_show_plane_segmentation_2d(self):
        color_wheel = ['red', 'blue', 'green', 'black', 'magenta', 'yellow']
        assert isinstance(show_plane_segmentation_2d(self.ps, color_by='pixel_mask', color_wheel=color_wheel), widgets.Widget)
        
    def test_plane_segmentation_2d_widget(self):
        assert isinstance(plane_segmentation_2d_widget(self.ps), widgets.Widget)
        
    def test_show_plane_segmentation_3d(self):
        assert isinstance(show_plane_segmentation_3d(self.ps), widgets.Widget)
        
    def test_show_plane_segmentation(self):
        assert isinstance(show_plane_segmentation(self.ps, default_neurodata_vis_spec), widgets.Widget)
        
    def test_show_image_segmentation(self):
        assert isinstance(show_image_segmentation(self.img_seg, default_neurodata_vis_spec), widgets.Widget)

