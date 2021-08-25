Use Case: Electrophysiology data
================================

This page shows how to open an nwbfile, retrieve various electrophysiology datsets and visualize them using nwb widgets in a jupyter notebook.

Using optical physiology data:
------------------------------


.. jupyter-execute::

    from datetime import datetime
    import ipywidgets as widgets
    import numpy as np
    from dateutil.tz import tzlocal
    from ndx_grayscalevolume import GrayscaleVolume
    from pynwb.device import Device
    from pynwb.ophys import (
        TwoPhotonSeries,
        OpticalChannel,
        ImageSegmentation,
        Fluorescence,
        DfOverF,
        ImagingPlane,
        PlaneSegmentation
    )

    from nwbwidgets.ophys import TwoPhotonSeriesWidget
    from nwbwidgets.ophys import (
        show_grayscale_volume,
        show_df_over_f,
        PlaneSegmentation2DWidget,
        show_plane_segmentation_3d_voxel,
        show_plane_segmentation_3d_mask,
        show_image_segmentation,
    )
    from nwbwidgets.view import default_neurodata_vis_spec


#. Displaying two photon series:

    #. Custom create a two photon series:

.. jupyter-execute::

    # create device and optical channel info:
    device = Device("imaging_device_1")
    optical_channel = OpticalChannel("my_optchan", "description", 500.0)
    # create a custom imaging plane:
    imaging_plane = ImagingPlane(
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

    image_series = TwoPhotonSeries(
        name="test_image_series",
        data=np.random.randn(100, 5, 5),
        imaging_plane=imaging_plane,
        starting_frame=[0],
        rate=1.0,
        unit="n.a",
    )
    img_seg = ImageSegmentation()
    ps2 = img_seg.create_plane_segmentation(
            "output from segmenting my favorite imaging plane",
            imaging_plane,
            "2d_plane_seg",
            image_series,
        )

    w, h = 3, 3
    img_mask1 = np.zeros((w, h))
    img_mask1[0, 0] = 1.1
    img_mask1[1, 1] = 1.2
    img_mask1[2, 2] = 1.3
    ps2.add_roi(image_mask=img_mask1)

    img_mask2 = np.zeros((w, h))
    img_mask2[0, 0] = 2.1
    img_mask2[1, 1] = 2.2
    ps2.add_roi(image_mask=img_mask2)

    img_mask2 = np.zeros((w, h))
    img_mask2[0, 0] = 9.1
    img_mask2[1, 1] = 10.2
    ps2.add_roi(image_mask=img_mask2)

    img_mask2 = np.zeros((w, h))
    img_mask2[0, 0] = 3.5
    img_mask2[1, 1] = 5.6
    ps2.add_roi(image_mask=img_mask2)

    fl = Fluorescence()
    rt_region = ps2.create_roi_table_region(
        "the first of two ROIs", region=[0, 1, 2, 3]
    )

    rois_shape = 5
    data = np.arange(10*rois_shape).reshape([10, -1], order='F')
    timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    rrs = fl.create_roi_response_series(
        name="my_rrs", data=data, rois=rt_region, unit="lumens", timestamps=timestamps
    )
    df_over_f = DfOverF(rrs)

    wid = TwoPhotonSeriesWidget(image_series, default_neurodata_vis_spec)
    display(wid)

.. jupyter-execute::

    # creating a widget for 3d images:
    image_series3 = TwoPhotonSeries(
        name="test_3d_images",
        data=np.random.randn(100, 5, 5, 5),
        imaging_plane=imaging_plane,
        starting_frame=[0],
        rate=1.0,
        unit="n.a",
    )
    wid = TwoPhotonSeriesWidget(image_series3, default_neurodata_vis_spec)
    display(wid)

    # displaying df over f traces:
    dff = show_df_over_f(df_over_f, default_neurodata_vis_spec)
    display(dff)


.. jupyter-execute::

    ps3 = PlaneSegmentation(
        "output from segmenting my favorite imaging plane",
        imaging_plane,
        "3d_plane_seg",
        image_series,
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
    display(wid)
