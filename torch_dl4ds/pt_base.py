"""
Base Training Class


"""

import os
import xarray as xr
import numpy as np
import torch as pt
from abc import ABC, abstractmethod


class TorchTrainer(ABC):
    """
    Base class for training Torch-based downscaling models.
    """

    def __init__(
        self,
        backbone,
        upsampling,
        data_train,
        loss='mae',
        batch_size=32,
        scale=4,
        device='cuda',
        verbose=True,
        save=True,
        save_path='./',
        show_plot=False,
    ):
        self.backbone = backbone
        self.upsampling = upsampling

        # Check training data format
        if not isinstance(data_train, (xr.DataArray, np.ndarray)):
            raise TypeError('`data_train` must be an xarray.DataArray or numpy.ndarray')

        if data_train.ndim != 4:
            raise ValueError('`data_train` must be 4D: [time, lat, lon, channels]')

        self.data_train = data_train
        self.loss = loss
        self.batch_size = batch_size
        self.scale = scale
        self.verbose = verbose
        self.save = save
        self.save_path = save_path if save_path.endswith('/') else save_path + '/'
        self.show_plot = show_plot

        # Device setup
        self.device = pt.device(device if pt.cuda.is_available() else 'cpu')

        # Scale check: ensure image size is divisible
        imsize = data_train.shape[2]  # assumes [T, C, H, W]
        if scale is not None and imsize % scale != 0:
            raise ValueError('Spatial size must be divisible by scale.')

        # Loss function
        if loss == 'mae':
            self.loss_fn = pt.nn.L1Loss()
        elif loss == 'mse':
            self.loss_fn = pt.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss}")

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def save_results(self, model, test_loss=None, runtime=None, val_losses=None):
        """
        Save the model, loss curve, test loss, and timing info.
        """
        if not self.save:
            return

        os.makedirs(self.save_path, exist_ok=True)

        # Save model weights only
        pt.save(model.state_dict(), os.path.join(self.save_path, 'model.pth'))

        # (Optional) Save full model including architecture
        # pt.save(model, os.path.join(self.save_path, 'full_model.pt'))
        
        # Save test loss
        if test_loss is not None:
            np.savetxt(os.path.join(self.save_path, 'test_loss.txt'), [test_loss])

        # Save runtime
        if runtime is not None:
            with open(os.path.join(self.save_path, 'runtime.txt'), 'w') as f:
                f.write(f"{runtime:.4f} seconds\n")

        # Save learning curve
        if val_losses is not None and len(val_losses) > 0:
            np.savetxt(os.path.join(self.save_path, 'val_loss_curve.txt'), val_losses)

        if self.verbose:
            print(f"[Saved] Model and results saved to: {self.save_path}")
