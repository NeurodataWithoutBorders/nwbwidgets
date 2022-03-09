from pathlib import Path
from typing import Union

try:
    import cv2

    HAVE_OPENCV = True
except ImportError:
    HAVE_OPENCV = False

try:
    from tifffile import imread, TiffFile

    HAVE_TIF = True
except ImportError:
    HAVE_TIF = False

PathType = Union[str, Path]

VIDEO_EXTENSIONS = [".mp4", ".avi", ".wmv", ".mov", ".flv"]


class VideoCaptureContext:
    """
    Context manager for opening videos using opencv
    """
    def __init__(self, video_path):
        self.vc = cv2.VideoCapture(video_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.vc.release()


def get_frame_shape(external_path_file: PathType):
    """
    Get frame shape
    Parameters
    ----------
    external_path_file: PathType
        path of external file from the external_file argument of ImageSeries
    """
    external_path_file = Path(external_path_file)
    if external_path_file.suffix in [".tif", ".tiff"]:
        return get_frame_shape_tif(external_path_file)
    elif external_path_file.suffix in VIDEO_EXTENSIONS:
        return get_frame_shape_video(external_path_file)
    else:
        raise NotImplementedError


def get_frame_count(external_path_file: PathType):
    """
    Get number of frames in the video or tif stack.
    Parameters
    ----------
    external_path_file: PathType
        path of external file from the external_file argument of ImageSeries
    """
    external_path_file = Path(external_path_file)
    if external_path_file.suffix in [".tif", ".tiff"]:
        return get_frame_count_tif(external_path_file)
    elif external_path_file.suffix in VIDEO_EXTENSIONS:
        return get_frame_count_video(external_path_file)
    else:
        raise NotImplementedError


def get_frame(external_path_file: PathType, index):
    """
    Get frame
    Parameters
    ----------
    external_path_file: PathType
        path of external file from the external_file argument of ImageSeries
    index: int
        the frame number to retrieve from the video/tif file
    """
    external_path_file = Path(external_path_file)
    if external_path_file.suffix in [".tif", ".tiff"]:
        return get_frame_tif(external_path_file, index)
    elif external_path_file.suffix in VIDEO_EXTENSIONS:
        return get_frame_video(external_path_file, index)
    else:
        raise NotImplementedError


def get_fps(external_path_file: PathType):
    external_path_file = Path(external_path_file)
    if external_path_file.suffix in [".tif", ".tiff"]:
        return get_fps_tif(external_path_file)
    elif external_path_file.suffix in VIDEO_EXTENSIONS:
        return get_fps_video(external_path_file)
    else:
        raise NotImplementedError

def get_frame_tif(external_path_file: PathType, index):
    external_path_file = Path(external_path_file)
    assert external_path_file.suffix in [".tif", ".tiff"], f"supply a tif file"
    assert HAVE_TIF, "pip install tifffile"
    return imread(str(external_path_file), key=int(index))


def get_frame_shape_tif(external_path_file: PathType):
    external_path_file = Path(external_path_file)
    assert external_path_file.suffix in [".tif", ".tiff"], f"supply a tif file"
    assert HAVE_TIF, "pip install tifffile"
    tif = TiffFile(external_path_file)
    page = tif.pages[0]
    return page.shape


def get_frame_count_tif(external_path_file: PathType):
    external_path_file = Path(external_path_file)
    assert external_path_file.suffix in [".tif", ".tiff"], f"supply a tif file"
    assert HAVE_TIF, "pip install tifffile"
    tif = TiffFile(external_path_file)
    return len(tif.pages)


def get_fps_tif(external_path_file: PathType):
    return


def get_frame_video(external_path_file: PathType, index):
    external_path_file = Path(external_path_file)
    assert (
        external_path_file.suffix in VIDEO_EXTENSIONS
    ), f"supply any of {VIDEO_EXTENSIONS} files"
    assert HAVE_OPENCV, "pip install opencv-python"
    no_frames = get_frame_count(external_path_file)
    assert index < no_frames, f"enter index < {no_frames}"
    if int(cv2.__version__.split(".")[0]) < 3:
        set_arg = cv2.cv.CV_CAP_PROP_POS_FRAMES
    else:
        set_arg = cv2.CAP_PROP_POS_FRAMES
    with VideoCaptureContext(str(external_path_file)) as cap:
        set_value = cap.vc.set(set_arg, index)
        success, frame = cap.vc.read()
    if success:
        return frame
    else:
        raise Exception("could not open video file")


def get_frame_count_video(external_path_file: PathType):
    external_path_file = Path(external_path_file)
    assert (
            external_path_file.suffix in VIDEO_EXTENSIONS
    ), f"supply any of {VIDEO_EXTENSIONS} files"
    assert HAVE_OPENCV, "pip install opencv-python"
    if int(cv2.__version__.split(".")[0]) < 3:
        frame_count_arg = cv2.cv.CV_CAP_PROP_FRAME_COUNT
    else:
        frame_count_arg = cv2.CAP_PROP_FRAME_COUNT
    with VideoCaptureContext(str(external_path_file)) as cap:
        frame_count = cap.vc.get(frame_count_arg)
    return frame_count


def get_frame_shape_video(external_path_file: PathType):
    external_path_file = Path(external_path_file)
    assert (
            external_path_file.suffix in VIDEO_EXTENSIONS
    ), f"supply any of {VIDEO_EXTENSIONS} files"
    assert HAVE_OPENCV, "pip install opencv-python"
    with VideoCaptureContext(str(external_path_file)) as cap:
        success, frame = cap.vc.read()
    if success:
        return frame.shape
    else:
        raise Exception("could not open video file")


def get_fps_video(external_path_file: PathType):
    external_path_file = Path(external_path_file)
    assert (
            external_path_file.suffix in VIDEO_EXTENSIONS
    ), f"supply any of {VIDEO_EXTENSIONS} files"
    assert HAVE_OPENCV, "pip install opencv-python"
    if int(cv2.__version__.split(".")[0]) < 3:
        fps_arg = cv2.cv.CV_CAP_PROP_FPS
    else:
        fps_arg = cv2.CAP_PROP_FPS
    with VideoCaptureContext(str(external_path_file)) as cap:
        fps = cap.vc.get(fps_arg)
    return fps