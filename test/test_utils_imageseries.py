from nwbwidgets.utils.imageseries import get_frame_count, get_frame, get_frame_shape
from .fixtures import create_movie_files, movie_shape, movie_no_frames, create_tif_files

def test_movie_frame(create_movie_files, movie_shape):
    frame = get_frame(create_movie_files[0], 0)
    assert frame.shape == movie_shape


def test_tif_frame(create_tif_files, movie_shape):
    frame = get_frame(str(create_tif_files[0]), 0)
    assert frame.shape == movie_shape


def test_movie_no_frames(create_movie_files, movie_no_frames):
    count = get_frame_count(create_movie_files[0])
    assert count == movie_no_frames[0]


def test_tif_no_frames(create_tif_files, movie_no_frames):
    count = get_frame_count(create_tif_files[0])
    assert count == movie_no_frames[0]


def test_movie_frame_shape(create_movie_files, movie_shape):
    shape = get_frame_shape(create_movie_files[0])
    assert shape == movie_shape


def test_tif_frame_shape(create_tif_files, movie_shape):
    shape = get_frame_shape(create_tif_files[0])
    assert shape == movie_shape
