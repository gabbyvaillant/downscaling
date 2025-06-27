import torch as pt
import torch.nn as nn
import numpy as np
import scipy as sc
import xarray as xr
import ecubevis as ecv
from torch.utils.data import Dataset

from .config import (
    POSTUPSAMPLING_METHODS
)
from .pt_utils import resize_array, checkarray_ndim

def create_pair_hr_lr(
    array,
    upsampling,
    scale,
    predictors=None,
    debug=False,
    interpolation='inter_area'):

    hr_array = array

    hr_y, hr_x = hr_array.shape[1:] #120, 120

    lr_y, lr_x = int(hr_y / scale), int(hr_x / scale) #40, 40

    if predictors is not None:
        if predictors is None or predictors.size == 0:
            raise ValueError(f"[Predictors Error] predictors is None or empty. shape={getattr(predictors, 'shape', 'unknown')}")
        if predictors.shape[1] != lr_y or predictors.shape[2] != lr_x:
            lr_array_predictors = resize_array(predictors, (lr_x, lr_y), squeezed=False)
            #print(f"predictors resized: {lr_array_predictors.shape}")
        else:
            lr_array_predictors = predictors

        # COARSENING HR ARRAY TO THE LR 
        lr_array = resize_array(hr_array, (lr_x, lr_y), squeezed=False)
        
        # Check dimensions
        hr_array = checkarray_ndim(hr_array, 3, -1)
        lr_array = checkarray_ndim(lr_array, 3, -1)
        lr_array_predictors = checkarray_ndim(lr_array_predictors, 3, -1)

        if predictors.shape[-1] == 0:
            raise ValueError("Your predictor array has no channels. This will cause model failure.")

        lr_array = np.concatenate([lr_array, lr_array_predictors], axis=0)

        if lr_array.shape[-1] == 0:
            raise ValueError(f"[Invalid input] lr_array has zero channels after concatenation")

    else:
        print(f"Calling resize_array on shape {hr_array.shape} with newsize = ({lr_x}, {lr_y})")
        lr_array = resize_array(hr_array, (lr_x, lr_y), squeezed=False)

        # Check dimensions
        hr_array = checkarray_ndim(hr_array, 3, -1)
        lr_array = checkarray_ndim(lr_array, 3, -1)

    hr_array = np.asarray(hr_array, dtype=np.float32)
    lr_array = np.asarray(lr_array, dtype=np.float32)

    #print(f'HR array: {hr_array.shape}, LR array: {lr_array.shape}')

    return (hr_array, lr_array)

def create_batch_hr_lr(
    all_indices,
    index,
    array,
    upsampling,
    scale=4,
    batch_size=32,
    predictors=None,
    interpolation='inter_area'):

    batch_rand_idx = all_indices[index * batch_size: (index + 1) * batch_size]

    batch_hr, batch_lr = [], []

    #print(f"array shape: {array.shape}")
    #print(f"length of array: {len(array)}")

    for i in batch_rand_idx:

        #array shape: (2697, 1, 120, 120)
        data_i = array[i]
        predictors_i = None if predictors is None else predictors[i]

        res = create_pair_hr_lr(
            array=data_i,
            upsampling=upsampling,
            scale=scale,
            interpolation=interpolation,
            predictors=predictors_i
        )

        hr_array, lr_array = res

        batch_hr.append(hr_array)
        batch_lr.append(lr_array)
    

    batch_hr = np.asarray(batch_hr)
    batch_lr = np.asarray(batch_lr)


    batch_hr = pt.from_numpy(batch_hr).float() # <class 'torch.Tensor'>
    batch_lr = pt.from_numpy(batch_lr).float() # <class 'torch.Tensor'>

    return batch_hr, batch_lr


class DataGenerator(Dataset):
    """
    DataGenerator creates batches of paired training samples for supervised learning.
    Designed for use with PyTorch-based spatial downscaling over time (1 sample = 1 time step).
    """

    def __init__(
        self,
        array,
        backbone,
        upsampling,
        scale,
        batch_size=32,
        predictors=None,
        interpolation='inter_area'
    ):

        # Load HR data
        if isinstance(array, xr.DataArray):
            self.array = array.values
        else:
            self.array = array

        # Other inputs
        self.batch_size = batch_size
        self.upsampling = upsampling
        self.scale = scale
        self.backbone = backbone

        # Combine predictors into one tensor (if we have more than one pred variable)
        if predictors is not None:
            self.predictors = np.concatenate(predictors, axis=-1)
        else:
            self.predictors = None

        self.interpolation = interpolation

        # Number of time steps (each is a separate sample)
        self.n = self.array.shape[0]

        # Generate shuffled indices
        #Example [0, 1, 2, ..., 2696]
        self.indices = np.random.permutation(np.arange(self.n))
        

    def __len__(self):
        """
        How long the dataset is. PyTorch's DataLoader calls this
        to know how many times to call __getitem__ per epoch.
        """
        #this way keeps all batches no matter the size
        #return (len(self.indices) + self.batch_size - 1) // self.batch_size
        #n_batches = self.n // self.batch_size

        #this way gets rid of the last batch that is exactly 32
        #return n_batches

        return self.n
    def __getitem__(self, index):
        """
        Loads one batch at index 
        """

        i = self.indices[index]
        data_i = self.array[i]
        predictors_i = None if self.predictors is None else self.predictors[i]

        hr_array, lr_array = create_pair_hr_lr(
            array=data_i,
            upsampling=self.upsampling,
            scale=self.scale,
            interpolation=self.interpolation,
            predictors=predictors_i
        )

        hr_tensor = pt.from_numpy(hr_array).float()
        lr_tensor = pt.from_numpy(lr_array).float()

        return lr_tensor, hr_tensor # (input, target)
