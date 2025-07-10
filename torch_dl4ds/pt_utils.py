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
import torch.nn.functional as F

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

##########################################################################################
def new_resize_array(array, newsize, squeezed=True):
    """
    Resize a 2D, 3D, or 4D array using PyTorch F.interpolate with area mode.
    Expects newsize = (W, H).

    Parameters
    ----------
    array : np.ndarray
    newsize : tuple (W, H)
    squeezed : bool

    Returns
    -------
    np.ndarray
    """
    # Ensure contiguous for torch conversion
    array = np.ascontiguousarray(array)

    # Determine input shape and reshape for PyTorch
    if array.ndim == 2:
        # (H, W)
        tensor = pt.tensor(array).unsqueeze(0).unsqueeze(0)  # (N=1, C=1, H, W)
    elif array.ndim == 3:
        # (C, H, W)
        tensor = pt.tensor(array).unsqueeze(0)  # (N=1, C, H, W)
    elif array.ndim == 4:
        # (T, C, H, W) --> treat T as batch dim
        tensor = pt.tensor(array)
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")

    # New size: (H, W)
    new_height = newsize[1]
    new_width = newsize[0]

    # Apply area-based downsampling
    resized = F.interpolate(
        tensor.float(),  # Ensure float
        size=(new_height, new_width),
        mode='area'
    )

    resized_np = resized.numpy()

    if array.ndim == 2:
        resized_np = resized_np.squeeze(axis=0).squeeze(axis=0)  # (H, W)
    elif array.ndim == 3 and resized_np.shape[0] == 1:
        resized_np = resized_np.squeeze(axis=0)  # (C, H, W)
    # For ndim==4, no squeeze — treat T as batch
    
    return np.squeeze(resized_np) if squeezed else resized_np

#######################################################################################

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

def plot_history(history, title=None, log_scale=False, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Validation Loss')

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title or 'Training History')
    ax.grid(True)
    ax.legend()

    if save_path:
        # Make sure the parent directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()
    plt.close(fig)

    return fig, ax