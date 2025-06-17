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
from .pt_utils import crop_array, resize_array, checkarray_ndim

def create_pair_hr_lr(
    array,
    array_lr,
    upsampling,
    scale,
    patch_size,
    static_vars=None,
    predictors=None,
    season=None,
    debug=False,
    interpolation='inter_area'):

    def preproc_static_vars(var):
        if patch_size is not None:
            var_hr = crop_array(np.squeeze(var), patch_size, yx=(crop_y, crop_x))
            var_hr = checkarray_ndim(var_hr, 3, -1)
            if upsampling in POSTUPSAMPLING_METHODS:
                var_lr = resize_array(var_hr, (patch_size_lr, patch_size_lr), interpolation)
            else:
                var_lr = var_hr
        else:
            var_hr = checkarray_ndim(var, 3, -1)
            if upsampling in POSTUPSAMPLING_METHODS:
                var_lr = resize_array(var, (lr_x, lr_y), interpolation)
            else:
                var_lr = var_hr
        var_lr = checkarray_ndim(var_lr, 3, -1)
        return var_hr, var_lr

    hr_array = array
    lr_is_given = array_lr is not None
    lr_array = array_lr if lr_is_given else None

    hr_y, hr_x = hr_array.shape[:2] if hr_array.ndim == 3 else hr_array.shape[1:3]
    lr_y, lr_x = (array_lr.shape[:2] if lr_is_given else (hr_y // scale, hr_x // scale))

    if patch_size is not None:
        patch_size_lr = patch_size // scale

    if predictors is not None:
        if predictors.shape[1] != lr_y or predictors.shape[2] != lr_x:
            lr_array_predictors = resize_array(predictors, (lr_x, lr_y), interpolation)
        else:
            lr_array_predictors = predictors

        if patch_size is not None:
            lr_array_predictors, crop_y, crop_x = crop_array(
                lr_array_predictors, patch_size_lr, yx=None, position=True
            )
            crop_y_hr, crop_x_hr = crop_y * scale, crop_x * scale
            hr_array = crop_array(np.squeeze(hr_array), patch_size, yx=(crop_y_hr, crop_x_hr))
            if lr_is_given:
                lr_array = crop_array(lr_array, patch_size_lr, yx=(crop_y, crop_x))

        if not lr_is_given:
            lr_array = resize_array(hr_array, (lr_x, lr_y), interpolation)

        hr_array = checkarray_ndim(hr_array, 3, -1)
        lr_array = checkarray_ndim(lr_array, 3, -1)
        lr_array_predictors = checkarray_ndim(lr_array_predictors, 3, -1)
        lr_array = np.concatenate([lr_array, lr_array_predictors], axis=-1)

    else:
        if patch_size is not None:
            if lr_is_given:
                lr_array, crop_y, crop_x = crop_array(lr_array, patch_size_lr, yx=None, position=True)
                crop_y_hr, crop_x_hr = crop_y * scale, crop_x * scale
                hr_array = crop_array(np.squeeze(hr_array), patch_size, yx=(crop_y_hr, crop_x_hr))
            else:
                hr_array, crop_y, crop_x = crop_array(hr_array, patch_size, yx=None, position=True)
                lr_array = resize_array(hr_array, (patch_size_lr, patch_size_lr), interpolation)
        elif not lr_is_given:
            lr_array = resize_array(hr_array, (lr_x, lr_y), interpolation)

        hr_array = checkarray_ndim(hr_array, 3, -1)
        lr_array = checkarray_ndim(lr_array, 3, -1)

    static_array_hr = []
    if static_vars is not None:
        for staticvar in static_vars:
            staticvar_hr, staticvar_lr = preproc_static_vars(staticvar)
            static_array_hr.append(staticvar_hr)
            lr_array = np.concatenate([lr_array, staticvar_lr], axis=-1)
        static_array_hr = np.concatenate(static_array_hr, axis=-1)

    if season is not None:
        size_hr = patch_size if patch_size is not None else hr_y
        size_lr = patch_size_lr if patch_size is not None else lr_y
        season_hr = _get_season_array_(season, size_hr, size_hr)
        season_lr = _get_season_array_(season, size_lr, size_lr)
        lr_array = np.concatenate([lr_array, season_lr], axis=-1)
        static_array_hr = np.concatenate([static_array_hr, season_hr], axis=-1) if static_array_hr != [] else season_hr

    hr_array = np.asarray(hr_array, dtype=np.float32)
    lr_array = np.asarray(lr_array, dtype=np.float32)
    static_array_hr = np.asarray(static_array_hr, dtype=np.float32) if static_array_hr != [] else None

    if debug:
        print(f'HR array: {hr_array.shape}, LR array: {lr_array.shape}')
        if static_array_hr is not None:
            print(f'Aux array HR: {static_array_hr.shape}')
        if patch_size is not None:
            print(f'Crop X,Y: {crop_x}, {crop_y}')

    return (hr_array, lr_array, static_array_hr) if static_array_hr is not None else (hr_array, lr_array)


def create_batch_hr_lr(
    all_indices,
    index,
    array,
    array_lr,
    upsampling,
    scale=4,
    batch_size=32,
    patch_size=None,
    time_window=None,
    static_vars=None,
    predictors=None,
    interpolation='inter_area',
    time_metadata=None):
    
    batch_rand_idx = all_indices[index * batch_size: (index + 1) * batch_size]
    batch_hr, batch_lr, batch_aux_hr = [], [], []

    for i in batch_rand_idx:
        data_i = array[i]
        data_lr_i = None if array_lr is None else array_lr[i]
        predictors_i = None if predictors is None else predictors[i]
        season_i = _get_season_(time_metadata[i]) if time_metadata is not None else None

        res = create_pair_hr_lr(
            array=data_i,
            array_lr=data_lr_i,
            upsampling=upsampling,
            scale=scale,
            patch_size=patch_size,
            static_vars=static_vars,
            season=season_i,
            interpolation=interpolation,
            predictors=predictors_i
        )

        if len(res) == 3:
            hr_array, lr_array, static_array_hr = res
            batch_aux_hr.append(static_array_hr)
        else:
            hr_array, lr_array = res

        batch_hr.append(hr_array)
        batch_lr.append(lr_array)

    batch_hr = np.asarray(batch_hr, dtype=np.float32)
    batch_lr = np.asarray(batch_lr, dtype=np.float32)
    
    if batch_aux_hr:
        batch_aux_hr = np.asarray(batch_aux_hr, dtype=np.float32)
        return [batch_lr, batch_aux_hr], [batch_hr]
    else:
        return [batch_lr], [batch_hr]


class DataGenerator(Dataset):
    """
    DataGenerator creates batches of paired training samples for supervised learning.
    Designed for use with PyTorch-based spatial downscaling over time (1 sample = 1 time step).
    """

    def __init__(
        self,
        array,
        array_lr,
        backbone,
        upsampling,
        scale,
        batch_size=32,
        patch_size=None,
        static_vars=None,
        predictors=None,
        interpolation='inter_area',
        repeat=None,
        time_metadata=None  # optional time info if seasonal encoding is used
    ):
        # Load HR data
        if isinstance(array, xr.DataArray):
            self.array = array.values
            if time_metadata is None and hasattr(array, "time"):
                self.time_metadata = array.time.copy()
            else:
                self.time_metadata = time_metadata
        else:
            self.array = array
            self.time_metadata = time_metadata

        # Load LR data
        self.array_lr = array_lr.values if isinstance(array_lr, xr.DataArray) else array_lr

        # Other inputs
        self.batch_size = batch_size
        self.scale = scale
        self.upsampling = upsampling
        self.backbone = backbone
        self.patch_size = patch_size
        self.static_vars = [
            v.values if isinstance(v, xr.DataArray) else v for v in static_vars
        ] if static_vars is not None else None

        # Combine predictors into one tensor
        if predictors is not None:
            self.predictors = np.concatenate(predictors, axis=-1)
        else:
            self.predictors = None

        self.interpolation = interpolation
        self.repeat = repeat

        # Number of time steps (each is a separate sample)
        self.n = self.array.shape[0]

        # Generate shuffled indices
        self.indices = np.random.permutation(np.arange(self.n))
        if self.repeat is not None and isinstance(self.repeat, int):
            self.indices = np.hstack([self.indices for _ in range(self.repeat)])

        # Validate patch size
        if patch_size is not None and self.upsampling in POSTUPSAMPLING_METHODS:
            if patch_size % scale != 0:
                raise ValueError("`patch_size` must be divisible by `scale`")

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        return create_batch_hr_lr(
            self.indices,
            index,
            self.array,
            self.array_lr,
            upsampling=self.upsampling,
            scale=self.scale,
            batch_size=self.batch_size,
            patch_size=self.patch_size,
            static_vars=self.static_vars,
            predictors=self.predictors,
            interpolation=self.interpolation,
            time_metadata=self.time_metadata
        )
    

def _get_season_(time_metadata, time_window=None):
    """Get the season for a given time index or range."""
    if time_window is None:
        month_int = time_metadata.dt.month.values
    else:
        from scipy import stats
        month_int = stats.mode(time_metadata.time.dt.month.values, keepdims=False).mode

    if month_int in [12, 1, 2]:
        return 'winter'
    elif month_int in [3, 4, 5]:
        return 'spring'
    elif month_int in [6, 7, 8]:
        return 'summer'
    elif month_int in [9, 10, 11]:
        return 'autumn'
    else:
        raise ValueError("Invalid month:", month_int)


def _get_season_array_(season, sizey, sizex):
    """Produce a one-hot encoded season array with shape (sizey, sizex, 4)."""
    if season not in ['winter', 'spring', 'summer', 'autumn']:
        raise ValueError('`season` not recognized')

    season_array = np.zeros((sizey, sizex, 4))
    idx = ['winter', 'spring', 'summer', 'autumn'].index(season)
    season_array[:, :, idx] = 1
    return season_array