"""
Training procedure for supervised models

"""

import os
import xarray as xr
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pt_base import TorchTrainer
from pt_utils import Timing
#from .. import POSTUPSAMPLING_METHODS
#from ..pt_models import net_postupsampling

class TorchSupervisedTrainer(TorchTrainer):
    """
    """
    def __init__(
            self,
            backbone,
            upsampling,
            data_train,
            data_val,
            data_test,
            data_train_lr=None,
            data_val_lr=None,
            data_test_lr=None,
            predictors_train=None,
            predictors_val=None,
            predictors_test=None,
            static_vars=None,
            scale=5,
            interpolation='inter_area',
            patch_size=None,
            batch_size=64,
            loss='mae',
            epochs=60,
            steps_per_epoch=None,
            test_steps=None,
            validation_steps=None,
            device='cuda',
            model_list=None,
            learning_rate=(1e-3, 1e-4),
            lr_decay_after=1e5,
            early_stopping=False,
            patience=6,
            min_delta=0,
            show_plot=True,
            save=False,
            save_path=None,
            save_bestmodel=False,
            trained_model=None,
            trained_epochs=0,
            verbose=True,
            **architecture_params
    ):
        """Training procedure for supervised models.
        
        Parameters
        ----------
        Fill in later
        """
        super().__init__(
        backbone=backbone,
        upsampling=upsampling,
        data_train=data_train,
        data_train_lr=data_train_lr,
        loss=loss,
        batch_size=batch_size,
        patch_size=patch_size,
        scale=scale,
        device=device,
        verbose=verbose,
        model_list=model_list,
        save=save,
        save_path=save_path,
        show_plot=show_plot
        )
        self.data_val = data_val
        self.data_test = data_test
        self.data_val_lr = data_val_lr
        self.data_test_lr = data_test_lr
        self.predictors_train = predictors_train
        if self.predictors_train is not None and not isinstance(self.predictors_train, list):
            raise TypeError('`predictors_train` must be a list of ndarrays')
        self.predictors_test = predictors_test
        if self.predictors_test is not None and not isinstance(self.predictors_test, list):
            raise TypeError('`predictors_test` must be a list of ndarrays')
        self.predictors_val = predictors_val
        if self.predictors_val is not None and not isinstance(self.predictors_val, list):
            raise TypeError('`predictors_val` must be a list of ndarrays')
        self.static_vars = static_vars 
        if self.static_vars is not None:
            for i in range(len(self.static_vars)):
                if isinstance(self.static_vars[i], xr.DataArray):
                    self.static_vars[i] = self.static_vars[i].values
        self.interpolation = interpolation 
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.test_steps = test_steps
        self.learning_rate = learning_rate
        self.lr_decay_after = lr_decay_after
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.show_plot = show_plot
        self.architecture_params = architecture_params
        self.trained_model = trained_model
        self.trained_epochs = trained_epochs
        self.save_bestmodel = save_bestmodel


    def setup_model(self):
        """
        Setting up the model
        """
        #I'm omitting any spatiotemporal stuff
        n_channels = self.data_train.shape[-1]
        n_aux_channels = 0
        if self.static_vars is not None:
            n_channels += len(self.static_vars)
            n_aux_channels = len(self.static_vars)
        if self.predictors_train is not None:
            n_channels += len(self.predictors_train)

        if self.patch_size is None:
            lr_height = int(self.data_train.shape[1] / self.scale)
            lr_width = int(self.data_train.shape[2] / self.scale)
            hr_height = int(self.data_train.shape[1])
            hr_width = int(self.data_train.shape[2])

        else:
            lr_height = lr_width = int(self.patch_size / self.scale)
            hr_height = hr_width = int(self.patch_size)

        ### Instantiating the model

        if self.trained_model is None:
            if self.upsampling in POSTUPSAMPLING_METHODS:
                self.model = net_postupsampling(
                    backbone_block=self.backbone,
                    upsampling=self.upsampling,
                    scale=self.scale,
                    lr_size=(lr_height, lr_width),
                    n_channels=n_channels,
                    n_aux_channels=n_aux_channels,
                    **self.architecture_params)
        else:
            self.model = self.trained_model
            print('Loading pre-trained model')

    def run(self):
        """
        Compiling, training and saving the model
        """
        self.timing = Timing(self.verbose)
        self.setup_model()

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate[0])
        self.criterion = nn.L1Loss() if self.loss == 'mae' else nn.MSELoss()

        #Need to implement the DataLoader class next?
        train_loader = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.ds_val, batch_size=self.batch_size)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.trained_epochs, self.epochs):
            self.model.train()
            total_train_loss = 0
            for batch in train_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            self.model.eval() #idk 
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = self.model(x)
                    loss = self.criterion(y_pred, y)
                    total_val_loss += loss.item()
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, "
                      f"Train Loss: {total_train_loss:.4f}, "
                      f"Val Loss: {total_val_loss:.4f}")
            
            #Early stopping logic
            if self.early_stopping:
                if total_val_loss < best_val_loss - self.min_delta:
                    best_val_loss = total_val_loss
                    patience_counter = 0
                    if self.save_bestmodel:
                        torch.save(self.model.state_dict(), os.path.join(self.save_path, "best_model.pt"))
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print("Early stopping triggered.")
                        break
        self.timing.runtime()

        if self.save:
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "final_model.pt"))


    

