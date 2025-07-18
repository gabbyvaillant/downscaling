import os
import time
from datetime import datetime
import torch as pt
import torch.nn as nn
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import math
import pandas as pd
import torch.nn.functional as F

from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates
)

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

class GPUMetricsLogger:
    def __init__(self, device_index=0, model=None):
        nvmlInit() #Need to call before any other pynvml functions
        self.handle = nvmlDeviceGetHandleByIndex(device_index) #reference to the GPU we are monitoring
        self.logs = [] #where we store the GPU usage logs
        self.start_time = time.time()
        self.model = model
        self.model_size = self.count_params(model) if model is not None else None
    
    def count_params(self, model):
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total

    def log(self, label=None):
        elapsed_time = time.time() - self.start_time
        mem_info = nvmlDeviceGetMemoryInfo(self.handle)
        util = nvmlDeviceGetUtilizationRates(self.handle)
    
        self.logs.append({
            'time_elapsed_sec': elapsed_time,
            'memory_used_GiB': mem_info.used / (1024 ** 3),
            'gpu_utilization_percent': util.gpu,
            'label': label
        })

    
    def to_dataframe(self):
        return pd.DataFrame(self.logs)

    
    def save_and_plot(self, save_path=None):
        df = self.to_dataframe()

        if self.model_size is not None:
            model_label = f"{self.model_size} params"
        else:
            model_label = "gpu_metrics"

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            csv_path = os.path.join(save_path, f"gpu_metrics_{self.model_size // 100000}M_params.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved GPU metrics to {csv_path}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
        fig.suptitle(f"GPU Metrics Over Time - Model: {model_label}", fontsize=16)
        
        # Memory
        axes[0].plot(df['time_elapsed_sec'], df['memory_used_GiB'], color='skyblue')
        axes[0].set_title("Memory Used (GiB)")
        axes[0].set_ylabel("GiB")
        axes[0].set_ylim(0, 4) # bc this VM has 4GB available
        axes[0].grid(True)
        
        # Utilization
        axes[1].plot(df['time_elapsed_sec'], df['gpu_utilization_percent'], color='salmon')
        axes[1].set_title("GPU Utilization (%)")
        axes[1].set_ylabel("%")
        axes[1].set_ylim(0, 100)
        axes[1].grid(True)

        # Third panel: Model Info
        axes[2].axis('off')  # hide axis
        
        info_text = f"""
        Batch Size: {self.model.batch_size if hasattr(self.model, 'batch_size') else 'N/A'}
        Parameters: {self.model_size // 1_000_000}M params
        Filters: {arch_params.get('n_filters', 'N/A')}
        Blocks: {arch_params.get('n_blocks', 'N/A')}
        Backbone: {self.model.__class__.__name__}
        Upsampling: {getattr(self.model, 'upsampling', 'N/A')}
        Device: {pt.cuda.get_device_name(0) if pt.cuda.is_available() else 'CPU'}
        """
        
        axes[2].text(0.05, 0.95, info_text.strip(), va='top', ha='left', fontsize=12, family='monospace')
        axes[2].set_title("Model Info")

        for ax in axes.flat:
            ax.set_xlabel("Time (sec)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plot_path = os.path.join(save_path, f"gpu_metrics_plot_{self.model_size // 1_000_000}M_params.png")
            plt.savefig(plot_path)
            print(f"Saved GPU plot to {plot_path}")
        plt.show()