import datetime
import time
import os
import numpy as np
import xarray as xr
import torch as pt
import torch.nn as nn
from .pt_utils import Timing, checkarray_ndim, resize_array


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
        predictors=None,
        interpolation='inter_area',
        batch_size=32,
        scaler=None,
        save_path=None,
        save_fname='y_hat.npy',
        return_lr=False,
        device='cpu'
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
        predictors : list of np.ndarray
            Dynamic predictor variables, shaped as [N, H, W, C].
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
            'cuda' or 'cpu'.
        """
        self.trainer = trainer
        self.array_in_hr = array_in_hr
        self.array = array
        self.scale = scale
        self.predictors = predictors
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
            predictors=self.predictors,
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
    predictors=None,
    interpolation='inter_area',
    batch_size=32,
    scaler=None,
    save_path=None,
    save_fname='y_hat.npy',
    return_lr=False,
    device='GPU'
):

    """Run inference on HR or LR data using a trained model."""
    timing = Timing()

    # Handle input
    if array_in_hr:
        raise ValueError("Inference expects LR input; set array_in_hr=False.")
        #Not sure if it should only take in LR .... idk why it would take in HR.. 


    def to_tensor(x, device='cpu'):
        if isinstance(x, pt.Tensor):
            return x.detach().clone().to(dtype=pt.float32, device=device)
        else:
            return pt.tensor(x, dtype=pt.float32, device=device)
    

    x_test_lr = to_tensor(array, device=device) # shape: (B, C, H, W)

    # Include predictors if provided
    if predictors is not None:
        x_predictors = to_tensor(predictors, device=device)
        x_input = pt.cat([x_test_lr, x_predictors], dim=1)
    else:
        x_input = x_test_lr

    trainer.eval()

    ## FIX THIS PART!!!!!!
    with pt.no_grad():
        y_pred = []
        for i in range(0, len(x_input), batch_size):
            batch = x_input[i:i + batch_size]
            output = trainer(batch)
            #if scaler is not None:
            #    output = scaler.inverse_transform(output)
            y_pred.append(output.cpu())

    y_pred = pt.cat(y_pred, dim=0).numpy()  # (B, C, H*scale, W*scale)

    # Optionally save
    if save_path and save_fname:
        np.save(os.path.join(save_path, save_fname), y_pred)

    return y_pred

