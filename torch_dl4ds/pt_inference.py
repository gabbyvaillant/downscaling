import datetime
import time
import os
import numpy as np
import xarray as xr
import torch as pt
import torch.nn as nn

from pt_utils import Timing, checkarray_ndim, resize_array, spatiotemporal_to_spatial_samples
from pt_dataloader import create_batch_hr_lr


class Predictor:
    """
    Predictor class for performing inference on unseen HR or LR data. The data
    (``array``) is super-resolved or downscaled using the trained super-resolution network (contained in ``trainer``)
    """

    def __init__(
        self,
        trainer,
        array,
        scale,
        array_in_hr=False,
        static_vars=None,
        predictors=None,
        time_window=None,
        time_metadata=None,
        interpolation='inter_area',
        batch_size=64,
        scaler=None,
        save_path=None,
        save_fname='y_hat.npy',
        return_lr=False,
        device='GPU'
    ):
        """
        Parameters
        ----------
        trainer : object
            Trainer containing a PyTorch model (trainer.model or trainer.generator).
        array : np.ndarray or xarray.DataArray
            HR or LR input data.
        scale : int
            Spatial scaling factor.
        array_in_hr : bool
            If True, assumes array is HR ground truth. Otherwise, assumes it's LR input.
        static_vars : list of 2D np.ndarray or xarray.DataArray
            Static spatial predictors.
        predictors : list of np.ndarray
            Dynamic predictor variables, shaped as [N, H, W, C].
        time_window : int or None
            Required if using a spatiotemporal model.
        time_metadata : list or None
            Optional timestamps to retrieve season or time-specific features.
        interpolation : str
            Interpolation method for resizing.
        batch_size : int
            Batch size for inference.
        scaler : object or None
            Optional scaler with inverse_transform().
        save_path : str or None
            If provided, output is saved to this path.
        save_fname : str
            Filename to save the output.
        return_lr : bool
            Whether to return the LR input used for prediction.
        device : str
            'GPU' or 'CPU'.
        """
        self.trainer = trainer
        self.array_in_hr = array_in_hr
        self.array = array
        self.scale = scale
        self.static_vars = static_vars
        self.predictors = predictors
        self.time_window = time_window
        self.time_metadata = time_metadata
        self.interpolation = interpolation
        self.batch_size = batch_size
        self.scaler = scaler
        self.save_path = save_path
        self.save_fname = save_fname
        self.return_lr = return_lr
        self.device = device

    def run(self):
        return predict(
            trainer=self.trainer,
            array=self.array,
            scale=self.scale,
            array_in_hr=self.array_in_hr,
            static_vars=self.static_vars,
            predictors=self.predictors,
            time_window=self.time_window,
            time_metadata=self.time_metadata,
            interpolation=self.interpolation,
            batch_size=self.batch_size,
            scaler=self.scaler,
            save_path=self.save_path,
            save_fname=self.save_fname,
            return_lr=self.return_lr,
            device=self.device
        )


def predict(
    trainer,
    array,
    scale,
    array_in_hr=True,
    static_vars=None,
    predictors=None,
    time_window=None,
    time_metadata=None,
    interpolation='inter_area',
    batch_size=64,
    scaler=None,
    save_path=None,
    save_fname='y_hat.npy',
    return_lr=False,
    device='GPU'
):
    """Run inference on HR or LR data using a trained model."""
    timing = Timing()

    # Select model
    if hasattr(trainer, 'model'):
        model = trainer.model
    elif hasattr(trainer, 'generator'):
        model = trainer.generator
    else:
        model = trainer

    # Setup device
    device = pt.device('cuda' if device == 'GPU' else 'cpu')
    model.to(device)
    model.eval()

    # Try to get upsampling mode for downstream use
    upsampling = getattr(trainer, 'upsampling', 'unknown')

    # Validate spatiotemporal model use
    dim = len(array.shape)
    if dim == 5 and time_window is None:
        raise ValueError('`time_window` must be provided for spatiotemporal models.')

    # Convert xarray to numpy
    if isinstance(array, xr.DataArray):
        array = array.values
    if static_vars is not None:
        for i in range(len(static_vars)):
            if isinstance(static_vars[i], xr.DataArray):
                static_vars[i] = static_vars[i].values

    n_samples = array.shape[0]
    if time_window is not None:
        n_samples -= time_window - 1

    # Merge predictor channels
    if predictors is not None:
        predictors = np.concatenate(predictors, axis=-1)

    # Prepare HR and LR arrays
    if array_in_hr:
        array_hr = array
        array_lr = None
    else:
        array = checkarray_ndim(array, 4, -1)
        hr_xy = (array.shape[2] * scale, array.shape[1] * scale)
        array_hr = resize_array(array, hr_xy, interpolation, squeezed=False)
        array_lr = array

    # Create batch
    batch = create_batch_hr_lr(
        all_indices=np.arange(n_samples),
        index=0,
        array=array_hr,
        array_lr=array_lr,
        upsampling=upsampling,
        scale=scale,
        batch_size=n_samples,
        patch_size=None,
        time_window=time_window,
        static_vars=static_vars,
        predictors=predictors,
        interpolation=interpolation,
        time_metadata=time_metadata
    )

    # Unpack inputs
    if static_vars is not None:
        [x_test_lr, batch_aux_hr], _ = batch
    else:
        [x_test_lr], _ = batch

    # Convert to PyTorch tensors
    x_test_lr = pt.tensor(x_test_lr, dtype=pt.float32).to(device)
    if static_vars is not None:
        batch_aux_hr = pt.tensor(batch_aux_hr, dtype=pt.float32).to(device)
        inputs = (x_test_lr, batch_aux_hr)
    else:
        inputs = x_test_lr

    # Run inference
    with pt.no_grad():
        out = model(*inputs) if isinstance(inputs, tuple) else model(inputs)

    # Postprocess
    out = out.cpu().numpy()
    if out.ndim == 5 and time_window is not None:
        out = spatiotemporal_to_spatial_samples(out, time_window)
    if scaler is not None:
        out = scaler.inverse_transform(out)
    if save_path is not None and save_fname is not None:
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, save_fname), out.astype('float32'))

    timing.runtime()

    return (out, x_test_lr.cpu().numpy()) if return_lr else out