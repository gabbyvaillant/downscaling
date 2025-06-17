"""
Base Training Class

TODO:
Add in more information to save_results
Import and translate the loss functions
"""

import os
import xarray as xr
import numpy as np
import torch as pt
from abc import ABC, abstractmethod
#from matplotlib.pyplot import show, close
#import logging

class TorchTrainer(ABC):
    """
    Base training class
    """
    def __init__(
        self,
        backbone,
        upsampling,
        data_train,
        data_train_lr=None,
        loss='mae',
        batch_size=64,
        patch_size=None,
        scale=4,
        device='cuda', #Use cuda for PT and GPU for TF
        verbose=True,
        save=True,
        save_path='./', #Saves to current working directory
        show_plot=False,
    ):
        #Checking training data split (both hr and lr)
        self.data_train = data_train
        if not isinstance(self.data_train, (xr.DataArray, np.ndarray)):
            msg = '`data_train` object must be of np.ndarray or xr.DataArray type'
            raise TypeError(msg)
        if not self.data_train.ndim >3:
            msg = '`data_train` must be 4D [samples, lat, lon, variables]'
            raise ValueError(msg)
        self.data_train_lr = data_train_lr
        if self.data_train_lr is not None:
            if not isinstance(self.data_train_lr, (xr.DataArray, np.ndarray)):
                msg = '`data_train_lr` must be a np.ndarray or xr.DataArray object'
                raise TypeError(msg)
            if self.data_train_lr.shape[0] != self.data_train.shape[0]:
                msg = '`data_train_lr` and `data_train` must contain '
                msg += 'the same number of samples (equal 1st dim length)'
                raise ValueError(msg)
            if not self.data_train_lr.ndim > 3:
                msg = '`data_train_lr` must be at least 4D [samples, lat, lon, variables]'
                raise ValueError(msg)
            
        self.device = pt.device(device if pt.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.loss = loss
        self.scale = scale
        self.verbose = verbose
        self.save = save
        self.save_path = save_path if save_path.endswith('/') else save_path + '/'
        self.show_plot = show_plot

        # Checking scale wrt image size
        #expecting: shape = (time, lat, lon, channel)
        if patch_size is not None:
            imsize = patch_size
        else:
            imsize = data_train.shape[-2] # We assume input shape [N, H, W, C] (i.e., xarray or numpy), not PyTorch tensor format

        
        if scale is not None and imsize % scale != 0:
            raise ValueError('Patch size must be divisible by scale')
        
        if data_train_lr is not None:
            scale_from_data = data_train.shape[-2] / data_train_lr.shape[-2]
            if not int(scale_from_data) == int(scale):
                raise ValueError('Scale mismatch between HR and LR data')

        # Choose loss function
        # loss = checkarg_loss(loss) #fill in this function
        #placeholder code:
        if loss == 'mae':
            self.loss_fn = pt.nn.L1Loss()
        elif loss == 'mse':
            self.loss_fn = pt.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss: {loss}")
        
    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def run(self):
        pass

    #TODO: finish adding in runtime, learning curve and anything extra to record
    def save_results(self, model, test_loss=None):
        """
        Save the torch model, learning curve, running time and test score.
        """
        if self.save:
            os.makedirs(self.save_path, exist_ok=True)
            pt.save(model.state_dict(), os.path.join(self.save_path, 'model.pth'))
            if test_loss is not None:
                np.savetxt(os.path.join(self.save_path, 'test_loss.txt'), [test_loss])
        
