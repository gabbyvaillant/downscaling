import os
import time
import xarray as xr
from datetime import datetime
import torch as pt
import torch.nn as nn
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import math
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .config import (
    BACKBONE_BLOCKS,
    DROPOUT_VARIANTS,
    LOSS_FUNCTIONS,
    UPSAMPLING_METHODS,
    INTERPOLATION_METHODS
)





def spatiotemporal_to_spatial_samples(array, time_window):
    """Remove dimension `time_window` from `array`, resulting in a sequence of
    spatial samples/grids. `time_window` is a dimension assumed to be in the 
    second place.

    ###TO-DO : other ways to collapse the time_window dimension
    """
    _, timew, _, _, _ = array.shape
    if timew != time_window:
        raise ValueError(
            '`time_window` must be located in the second position [n_samples, time_window, lat, lon, vars]')
    array_out = array[:, 0, :, :, :]
    array_out = np.concatenate([array_out, array[-1, 1:, :, :, :]], axis=0)
    return array_out

def checkarray_ndim(array, ndim=3, add_axis_position=-1):
    """Check the np.ndarray has at least `ndim` dimensions. If needed a new
    dimension (of lenght 1) is added at the position given by `add_axis_position`.
    """
    if array.ndim < ndim:
        return np.expand_dims(array, axis=add_axis_position)
    else:
        return array

def checkarg_upsampling(upsampling):
    """Check the argument ``upsampling``.

    Parameters
    ----------
    upsampling : str
        Upsampling method. 
    """ 
    #check if it is a string
    if not isinstance(upsampling, str):
        raise TypeError('`upsampling` must be a string')
    
    #check if it is a valid upsampling method
    if upsampling not in UPSAMPLING_METHODS:
        msg = f'`upsampling` is not recognized. Must be one of the '
        msg += f'following: {UPSAMPLING_METHODS}. Got {upsampling}'
        raise ValueError(msg)
    else:
        return upsampling


def checkarg_backbone(backbone):
    """Check the argument ``backbone``.

    Parameters
    ----------
    backbone : str
        Backbone block. 
    """ 
    if not isinstance(backbone, str):
        raise TypeError('`backbone` must be a string')

    if backbone not in BACKBONE_BLOCKS:
        msg = f'`backbone` not recognized. Must be one of the '
        msg += f'following: {BACKBONE_BLOCKS}. Got {backbone}'
        raise ValueError(msg)
    else:
        return backbone
    
def checkarg_dropout_variant(dropout_variant):
    """Check the argument ``dropout_variant``.

    Parameters
    ----------
    dropout_variant : str
        Desired dropout variant.  
    """

    if dropout_variant is None or dropout_variant == 'vanilla':
        return dropout_variant
    elif isinstance(dropout_variant, str):
        if dropout_variant not in DROPOUT_VARIANTS:
            msg = f"`dropout_variant must be None or one of {DROPOUT_VARIANTS}, got {dropout_variant}"
            raise ValueError(msg)
        else:
            return dropout_variant 


def crop_array(array, size, yx=None, position=False, exclude_borders=False,
               get_copy=False):
    """ 
    Return a squared cropped version of a 2D, 3D, or 4D or 5D

    Parameters
    __________

    array : numpy ndarray
    size : int
        Size of the cropped image.
    yx : tuple of int or None, optional
        Y,X coordinate of the bottom-left corner. If None then a random
        position will be chosen.
    position : bool, optional
        If set to True return also the coordinates of the bottom-left corner.
    get_copy : bool, optional
        If True a cropped copy of the intial array is returned. By default a
        sliced view of the array is returned.
    
    Returns
    -------
    cropped_array : numpy ndarray
        Cropped ndarray. By default a view is returned, unless ``get_copy``
        is True. 
    y, x : int
        [position=True] Y,X coordinates.
    """
    if array.ndim not in [2, 3, 4, 5]:
        raise TypeError('Input array is not a 2D, 3D, or 4D ndarray')
    if not isinstance(size, int):
        raise TypeError('`Size` must be integer')
    if array.ndim in [2, 3]:
        # assuming 3D ndarray as multichannel grid [lat, lon, vars] or [y, x, channels]
        array_size_y = array.shape[0]
        array_size_x = array.shape[1]
    elif array.ndim == 4:
        # assuming 4D ndarray [aux dim, lat, lon, vars] or [time, y, x, channels]
        array_size_y = array.shape[1]
        array_size_x = array.shape[2]
    elif array.ndim == 5:
        # assuming 4D ndarray [aux dim, time, y, x, channels]
        array_size_y = array.shape[2]
        array_size_x = array.shape[3]
    if size > array_size_y or size > array_size_x: 
        msg = "`Size` larger than the input image size"
        raise ValueError(msg)

    if yx is not None and isinstance(yx, tuple):
        y, x = yx
    else:
        # random location
        if exclude_borders:
            y = np.random.randint(1, array_size_y - size - 1)
            x = np.random.randint(1, array_size_x - size - 1)
        else:
            y = np.random.randint(0, array_size_y - size)
            x = np.random.randint(0, array_size_x - size)

    y0, y1 = y, int(y + size)
    x0, x1 = x, int(x + size)

    if y0 < 0 or x0 < 0 or y1 > array_size_y or x1 > array_size_x:
        raise RuntimeError(f'Cropped image cannot be obtained with size={size}, y={y}, x={x}')

    if get_copy:
        if array.ndim == 2:
            cropped_array = array[y0: y1, x0: x1].copy()
        elif array.ndim == 3:
            cropped_array = array[y0: y1, x0: x1, :].copy()
        elif array.ndim == 4:
            cropped_array = array[:, y0: y1, x0: x1, :].copy()
        elif array.ndim == 5:
            cropped_array = array[:, :, y0: y1, x0: x1, :].copy()
    else:
        if array.ndim == 2:
            cropped_array = array[y0: y1, x0: x1]
        elif array.ndim == 3:
            cropped_array = array[y0: y1, x0: x1, :]
        elif array.ndim == 4:
            cropped_array = array[:, y0: y1, x0: x1, :]
        elif array.ndim == 5:
            cropped_array = array[:, :, y0: y1, x0: x1, :]

    if position:
        return cropped_array, y, x
    else:
        return cropped_array

def resize_array(array, newsize, squeezed=True):
    """
    Resize a 2D, 3D, or 4D array using scipy.ndimage.zoom instead of cv2.

    Parameters
    ----------
    array : np.ndarray
        Input array with shape (y, x), (y, x, c), or (t, y, x, c).
    newsize : tuple
        Target size (width, height).
    squeezed : bool
        Whether to squeeze output dimensions of size 1.

    Returns
    -------
    np.ndarray
        Resized array.
    """
    if array.ndim == 2:
        zoom_factors = [newsize[1] / array.shape[0], newsize[0] / array.shape[1]]
        resized = zoom(array, zoom_factors, order=1)  # bilinear
    elif array.ndim == 3:
        zoom_factors = [newsize[1] / array.shape[0], newsize[0] / array.shape[1], 1]
        resized = zoom(array, zoom_factors, order=1)
    elif array.ndim == 4:
        t, y, x, c = array.shape
        zoom_factors = [1, newsize[1] / y, newsize[0] / x, 1]
        resized = zoom(array, zoom_factors, order=1)
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")

    return np.squeeze(resized) if squeezed else resized

class Timing:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.start_time = time.time()

    def runtime(self):
        end_time = time.time()
        duration = end_time = self.start_time
        if self.verbose:
            print(f"Total runtime: {duration:.2f} seconds")

