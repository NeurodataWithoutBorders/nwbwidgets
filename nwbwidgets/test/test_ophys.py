import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile
import ipywidgets as widgets
from ndx_grayscalevolume import GrayscaleVolume
from nwbwidgets.view import default_neurodata_vis_spec
from pynwb.ophys import TwoPhotonSeries, OpticalChannel, ImageSegmentation, Fluorescence, DfOverF
from pynwb.device import Device
from nwbwidgets.ophys import show_grayscale_volume,show_two_photon_series,show_df_over_f,show_plane_segmentation_2d
import unittest



def test_show_grayscale_volume():
    vol = GrayscaleVolume(name='vol',data=np.random.rand(2700).reshape((30,30,3)))
    assert isinstance(show_grayscale_volume(vol, default_neurodata_vis_spec),widgets.Widget)
    
    
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
        imaging_plane = nwbfile.create_imaging_plane('my_imgpln', optical_channel, 'a very interesting part of the brain',
                                                     device, 600., 300., 'GFP', 'my favorite brain location',
                                                     np.ones((5, 5, 3)), 4.0, 'manifold unit', 'A frame to refer to')

        self.image_series = TwoPhotonSeries(name='test_iS', dimension=[2],data=np.random.rand(10, 5, 5, 3),
                                       external_file=['images.tiff'], imaging_plane=imaging_plane,
                                       starting_frame=[0], format='tiff', starting_time=0.0, rate=1.0)
        nwbfile.add_acquisition(self.image_series)


        mod = nwbfile.create_processing_module('ophys', 'contains optical physiology processed data')
        img_seg = ImageSegmentation()
        mod.add(img_seg)
        self.ps = img_seg.create_plane_segmentation('output from segmenting my favorite imaging plane',
                                               imaging_plane, 'my_planeseg', self.image_series)


        w, h = 3, 3
        pix_mask1 = [(0, 0, 1.1), (1, 1, 1.2), (2, 2, 1.3)]
        vox_mask1 = [(0, 0, 0, 1.1), (1, 1, 1, 1.2), (2, 2, 2, 1.3)]
        img_mask1 = [[0.0 for x in range(w)] for y in range(h)]
        img_mask1[0][0] = 1.1
        img_mask1[1][1] = 1.2
        img_mask1[2][2] = 1.3
        self.ps.add_roi(pixel_mask=pix_mask1, image_mask=img_mask1, voxel_mask=vox_mask1)

        pix_mask2 = [(0, 0, 2.1), (1, 1, 2.2)]
        vox_mask2 = [(0, 0, 0, 2.1), (1, 1, 1, 2.2)]
        img_mask2 = [[0.0 for x in range(w)] for y in range(h)]
        img_mask2[0][0] = 2.1
        img_mask2[1][1] = 2.2
        self.ps.add_roi(pixel_mask=pix_mask2, image_mask=img_mask2, voxel_mask=vox_mask2)

        fl = Fluorescence()
        mod.add(fl)

        rt_region = self.ps.create_roi_table_region('the first of two ROIs', region=[0])

        data = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]).reshape(10,1)
        timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rrs = fl.create_roi_response_series('my_rrs', data, rt_region, unit='lumens', timestamps=timestamps)

        self.df_over_f = DfOverF(rrs)
        
    def test_show_two_photon_series(self):
        assert isinstance(show_two_photon_series(self.image_series,default_neurodata_vis_spec),widgets.Widget)
        
    def test_show_df_over_f(self):
        assert isinstance(show_df_over_f(self.df_over_f,default_neurodata_vis_spec),widgets.Widget)
        
    def test_show_plane_segmentation_2d(self):
        color_wheel = ['red', 'blue', 'green', 'black', 'magenta', 'yellow']
        assert isinstance(show_plane_segmentation_2d(self.ps,color_by='pixel_mask',color_wheel=color_wheel),plt.Figure)

