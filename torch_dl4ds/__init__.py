__version__ = "1.0"

BACKBONE_BLOCKS = [
    'resnet',           # residual convolutional blocks
    'densenet',         # dense convolutional blocks
]

UPSAMPLING_METHODS = [
    'spc',              # pixel shuffle or subpixel convolution in post-upscaling
]

POSTUPSAMPLING_METHODS = ['spc']

INTERPOLATION_METHODS = [
    'inter_area',       # resampling using pixel area relation (from opencv)
]

LOSS_FUNCTIONS = [
    'mae',              # mean absolute error  
    'mse',              # mean squarred error  
]

DROPOUT_VARIANTS = [
    'vanilla',          # vanilla dropout
    'gaussian',         # gaussian dropout
    'spatial',          # spatial dropout
    'mcdrop',           # monte carlo (vanilla) dropout
    'mcgaussiandrop',   # monte carlo gaussian dropout
    'mcspatialdrop']    # monte carlo spatial dropout

from .pt_metrics import *
from .pt_inference import *
from .pt_utils import *
from .pt_dataloader import *
from .pt_preprocessing import *
from .pt_base import *
from .pt_blocks import *
from .pt_losses import *
from .pt_postups import *
from .pt_supervised import *
