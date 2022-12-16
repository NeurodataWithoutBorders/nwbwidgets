"""
Taken from https://github.com/maartenbreddels/ipyvolume/pull/178
"""
import matplotlib.cm
import matplotlib.colors
import numpy as np
from ipyvolume import TransferFunction


def linear_transfer_function(color, min_opacity=0, max_opacity=0.05, reverse_opacity=False, n_elements=256):
    """Transfer function of a single color and linear opacity.
    :param color: List-like RGB, or string with hexadecimal or named color.
        RGB values should be within 0-1 range.
    :param min_opacity: Minimum opacity, default value is 0.0.
        Lowest possible value is 0.0, optional.
    :param max_opacity: Maximum opacity, default value is 0.05.
        Highest possible value is 1.0, optional.
    :param reverse_opacity: Linearly decrease opacity, optional.
    :param n_elements: Length of rgba array transfer function attribute.
    :type color: list-like or string
    :type min_opacity: float, int
    :type max_opacity: float, int
    :type reverse_opacity: bool
    :type n_elements: int
    :return: transfer_function
    :rtype: ipyvolume TransferFunction
    :Example:
    >>> import ipyvolume as ipv
    >>> green_tf = ipv.transfer_function.linear_transfer_function('green')
    >>> ds = ipv.datasets.aquariusA2.fetch()
    >>> ipv.volshow(ds.data[::4,::4,::4], tf=green_tf)
    >>> ipv.show()
    .. seealso:: matplotlib_transfer_function()
    """
    r, g, b = matplotlib.colors.to_rgb(color)
    opacity = np.linspace(min_opacity, max_opacity, num=n_elements)
    if reverse_opacity:
        opacity = np.flip(opacity, axis=0)
    rgba = np.transpose(np.stack([[r] * n_elements, [g] * n_elements, [b] * n_elements, opacity]))
    transfer_function = TransferFunction(rgba=rgba)
    return transfer_function


def matplotlib_transfer_function(
    colormap_name,
    min_opacity=0,
    max_opacity=0.05,
    reverse_colormap=False,
    reverse_opacity=False,
    n_elements=256,
):
    """Transfer function from matplotlib colormaps.
    :param colormap_name: name of matplotlib colormap
    :param min_opacity: Minimum opacity, default value is 0.
        Lowest possible value is 0, optional.
    :param max_opacity: Maximum opacity, default value is 0.05.
        Highest possible value is 1.0, optional.
    :param reverse_colormap: reversed matplotlib colormap, optional.
    :param reverse_opacity: Linearly decrease opacity, optional.
    :param n_elements: Length of rgba array transfer function attribute.
    :type colormap_name: str
    :type min_opacity: float, int
    :type max_opacity: float, int
    :type reverse_colormap: bool
    :type reverse_opacity: bool
    :type n_elements: int
    :return: transfer_function
    :rtype: ipyvolume TransferFunction
    :Example:
    >>> import ipyvolume as ipv
    >>> rgb = (0, 255, 0)  # RGB value for green
    >>> green_tf = ipv.transfer_function.matplotlib_transfer_function('bone')
    >>> ds = ipv.datasets.aquariusA2.fetch()
    >>> ipv.volshow(ds.data[::4,::4,::4], tf=green_tf)
    >>> ipv.show()
    .. seealso:: linear_transfer_function()
    """
    cmap = matplotlib.cm.get_cmap(name=colormap_name)
    rgba = np.array([cmap(i) for i in np.linspace(0, 1, n_elements)])
    if reverse_colormap:
        rgba = np.flip(rgba, axis=0)
    # Create opacity values to overwrite default matplotlib opacity=1.0
    opacity = np.linspace(min_opacity, max_opacity, num=n_elements)
    if reverse_opacity:
        opacity = np.flip(opacity, axis=0)
    rgba[:, -1] = opacity  # replace opacity=1 with actual opacity
    transfer_function = TransferFunction(rgba=rgba)
    return transfer_function
