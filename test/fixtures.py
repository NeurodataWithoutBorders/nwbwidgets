import cv2
import numpy as np
import pytest
from tifffile import imwrite


@pytest.fixture(scope="session")
def movie_fps():
    return 10


@pytest.fixture(scope="session")
def movie_shape():
    return (30, 40, 3)


@pytest.fixture(scope="session")
def movie_no_frames():
    return 10, 15


@pytest.fixture(scope="session")
def create_frames(movie_no_frames, movie_shape):
    mov_ar1 = np.random.randint(0, 255, size=[*movie_shape, movie_no_frames[0]], dtype="uint8")
    mov_ar2 = np.random.randint(0, 255, size=[*movie_shape, movie_no_frames[1]], dtype="uint8")
    return mov_ar1, mov_ar2


@pytest.fixture(scope="session")
def create_movie_files(tmp_path_factory, create_frames, movie_fps):
    base_path = tmp_path_factory.mktemp('moviefiles')
    print(base_path)
    mov_ar1_path = base_path/'movie1.avi'
    mov_ar2_path = base_path/'movie.avi'
    mov_array1, mov_array2 = create_frames
    movie_shape = mov_array1.shape[:2]
    no_frames = mov_array1.shape[-1]
    cap_mp4 = cv2.VideoWriter(str(mov_ar1_path),
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              movie_fps, movie_shape)
    cap_avi = cv2.VideoWriter(str(mov_ar2_path),
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              movie_fps, movie_shape)
    for frame_no in range(no_frames):
        cap_mp4.write(mov_array1[:,:,:,frame_no])
        cap_avi.write(mov_array2[:,:,:,frame_no])

    cap_mp4.release()
    cap_avi.release()
    return mov_ar2_path, mov_ar2_path


@pytest.fixture(scope="session")
def create_tif_files(tmp_path_factory, create_frames):
    base_path = tmp_path_factory.mktemp('tiffiles')
    print(base_path)
    frame1 = create_frames[0][:, :, :, 0]
    frame2 = create_frames[1][:, :, :, 0]
    tif_path1 = base_path/'tif_image1.tif'
    tif_path2 = base_path/'tif_image2.tif'
    imwrite(str(tif_path1), frame1, photometric='rgb')
    imwrite(str(tif_path2), frame2, photometric='rgb')
    return tif_path1, tif_path2
