import unittest

import ipywidgets as widgets
import numpy as np
from ndx_grayscalevolume import GrayscaleVolume
from pynwb.device import Device
from pynwb.ophys import (
    DfOverF,
    Fluorescence,
    ImageSegmentation,
    ImagingPlane,
    OpticalChannel,
    PlaneSegmentation,
    TwoPhotonSeries,
)
from pynwb.testing.mock.ophys import mock_PlaneSegmentation

from nwbwidgets.ophys import (
    PlaneSegmentation2DWidget,
    TwoPhotonSeriesWidget,
    show_df_over_f,
    show_grayscale_volume,
    show_image_segmentation,
    show_plane_segmentation_3d_mask,
    show_plane_segmentation_3d_voxel,
)
from nwbwidgets.view import default_neurodata_vis_spec


def test_show_grayscale_volume():
    vol = GrayscaleVolume(name="vol", data=np.random.rand(2700).reshape((30, 30, 3)))
    assert isinstance(show_grayscale_volume(vol, default_neurodata_vis_spec), widgets.Widget)


class CalciumImagingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        device = Device("imaging_device_1")
        optical_channel = OpticalChannel("my_optchan", "description", 500.0)
        self.imaging_plane = ImagingPlane(
            name="imgpln1",
            optical_channel=optical_channel,
            description="a fake ImagingPlane",
            device=device,
            excitation_lambda=600.0,
            imaging_rate=300.0,
            indicator="GFP",
            location="somewhere in the brain",
            reference_frame="unknown",
            origin_coords=[10, 20],
            origin_coords_unit="millimeters",
            grid_spacing=[0.001, 0.001],
            grid_spacing_unit="millimeters",
        )

        self.image_series = TwoPhotonSeries(
            name="test_image_series",
            data=np.random.randn(100, 5, 5),
            imaging_plane=self.imaging_plane,
            starting_frame=[0],
            rate=1.0,
            unit="n.a",
        )
        self.img_seg = ImageSegmentation()
        self.ps2 = self.img_seg.create_plane_segmentation(
            "output from segmenting my favorite imaging plane",
            self.imaging_plane,
            "2d_plane_seg",
            self.image_series,
        )
        self.ps2.add_column("type", "desc")
        self.ps2.add_column("type2", "desc")

        w, h = 3, 3
        img_mask1 = np.zeros((w, h))
        img_mask1[0, 0] = 1.1
        img_mask1[1, 1] = 1.2
        img_mask1[2, 2] = 1.3
        self.ps2.add_roi(image_mask=img_mask1, type=1, type2=0)

        img_mask2 = np.zeros((w, h))
        img_mask2[0, 0] = 2.1
        img_mask2[1, 1] = 2.2
        self.ps2.add_roi(image_mask=img_mask2, type=1, type2=1)

        img_mask2 = np.zeros((w, h))
        img_mask2[0, 0] = 9.1
        img_mask2[1, 1] = 10.2
        self.ps2.add_roi(image_mask=img_mask2, type=2, type2=0)

        img_mask2 = np.zeros((w, h))
        img_mask2[0, 0] = 3.5
        img_mask2[1, 1] = 5.6
        self.ps2.add_roi(image_mask=img_mask2, type=2, type2=1)

        fl = Fluorescence()
        rt_region = self.ps2.create_roi_table_region("the first of two ROIs", region=[0, 1, 2, 3])

        rois_shape = 5
        data = np.arange(10 * rois_shape).reshape([10, -1], order="F")
        timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        rrs = fl.create_roi_response_series(
            name="my_rrs", data=data, rois=rt_region, unit="lumens", timestamps=timestamps
        )
        self.df_over_f = DfOverF(rrs)

    def test_show_two_photon_series(self):
        wid = TwoPhotonSeriesWidget(self.image_series, default_neurodata_vis_spec)
        assert isinstance(wid, widgets.Widget)
        wid.controls["slider"].value = 50

    def test_show_3d_two_photon_series(self):
        image_series3 = TwoPhotonSeries(
            name="test_3d_images",
            data=np.random.randn(100, 5, 5, 5),
            imaging_plane=self.imaging_plane,
            starting_frame=[0],
            rate=1.0,
            unit="n.a",
        )
        wid = TwoPhotonSeriesWidget(image_series3, default_neurodata_vis_spec)
        assert isinstance(wid, widgets.Widget)
        wid.controls["slider"].value = 50

    def test_show_df_over_f(self):
        dff = show_df_over_f(self.df_over_f, default_neurodata_vis_spec)
        assert isinstance(dff, widgets.Widget)
        dff.controls["gas"].window = [1, 2]

    def test_plane_segmentation_2d_widget(self):
        wid = PlaneSegmentation2DWidget(self.ps2)
        assert isinstance(wid, widgets.Widget)
        wid.button.click()
        wid.cat_controller.value = "type"
        wid.cat_controller.value = "type2"

    def test_show_plane_segmentation_3d_mask(self):
        ps3 = PlaneSegmentation(
            "output from segmenting my favorite imaging plane",
            self.imaging_plane,
            "3d_plane_seg",
            self.image_series,
        )

        w, h, d = 3, 3, 3
        img_mask1 = np.zeros((w, h, d))
        for i in range(3):
            img_mask1[i, i, i] = 1.0
        ps3.add_roi(image_mask=img_mask1)

        img_mask2 = np.zeros((w, h, d))
        for i in range(3):
            img_mask2[i, i, i] = 1.2
        ps3.add_roi(image_mask=img_mask2)
        wid = show_plane_segmentation_3d_mask(ps3)
        assert isinstance(wid, widgets.Widget)

    def test_show_plane_segmentation_3d_voxel(self):
        ps3v = PlaneSegmentation(
            "output from segmenting my favorite imaging plane",
            self.imaging_plane,
            "3d_voxel",
            self.image_series,
        )

        voxel_mask = [(i, i, i, 1.0) for i in range(3)]
        ps3v.add_roi(voxel_mask=voxel_mask)

        voxel_mask = [(1, 1, i, 1.2) for i in range(3)]
        ps3v.add_roi(voxel_mask=voxel_mask)
        wid = show_plane_segmentation_3d_voxel(ps3v)
        assert isinstance(wid, widgets.Widget)

    def test_show_image_segmentation(self):
        assert isinstance(
            show_image_segmentation(self.img_seg, default_neurodata_vis_spec),
            widgets.Widget,
        )


def test_plane_segmentation_many_categories():

    ps = mock_PlaneSegmentation(n_rois=0)
    ps.add_column("category", "category")

    for i in range(50):
        image_mask = np.zeros((100, 100))

        # randomly generate example image masks
        x = np.random.randint(0, 95)
        y = np.random.randint(0, 95)
        image_mask[x : x + 5, y : y + 5] = 1

        # add image mask to plane segmentation
        ps.add_roi(
            image_mask=image_mask,
            category=str(i % 17),
        )

    PlaneSegmentation2DWidget(ps)
