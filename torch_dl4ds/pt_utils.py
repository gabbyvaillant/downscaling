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


def resize_array(array, newsize, squeezed=True):
    """
    Resize a 2D, 3D, or 4D array using scipy.ndimage.zoom.
    Expects newsize = (width, height)

    Parameters
    ----------
    array : np.ndarray
    newsize : tuple (W, H)
    squeezed : bool

    Returns
    -------
    np.ndarray
    """
    array = np.ascontiguousarray(array)

    if array.ndim == 2:
        zoom_factors = [newsize[1] / array.shape[0], newsize[0] / array.shape[1]]
    elif array.ndim == 3:
        # (C, H, W)
        zoom_factors = [1, newsize[1] / array.shape[1], newsize[0] / array.shape[2]]
    elif array.ndim == 4:
        # (T, C, H, W)
        t, c, h, w = array.shape
        zoom_factors = [1, 1, newsize[1] / h, newsize[0] / w]
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")


    resized = zoom(array, zoom_factors, order=1)
    return np.squeeze(resized) if squeezed else resized

class Timing:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.start = time.time()

    def runtime(self):
        end = time.time()
        elapsed = end - self.start
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        if self.verbose:
            print(f"Total runtime: {hours}h {minutes}m {seconds}s")
        return elapsed

