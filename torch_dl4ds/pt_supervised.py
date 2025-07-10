"""
Training procedure for supervised models

"""

import os
import xarray as xr
import torch.nn as nn
import torch
import numpy as np
import time
import datetime
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from .pt_base import TorchTrainer
from .pt_utils import (Timing, plot_history)
from .config import (
    POSTUPSAMPLING_METHODS
)
import matplotlib.pyplot as plt
from .pt_postups import net_postupsampling
from .pt_dataloader import DataGenerator


class TorchSupervisedTrainer(TorchTrainer):
    def __init__(
            self,
            backbone,
            upsampling,
            data_train,
            data_val,
            data_test,
            predictors_train=None,
            predictors_val=None,
            predictors_test=None,
            scale=5,
            interpolation='inter_area',
            batch_size=32,
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
        loss=loss,
        batch_size=batch_size,
        scale=scale,
        device=device,
        verbose=verbose,
        save=save,
        save_path=save_path,
        show_plot=show_plot
        )

        self.backbone = backbone
        self.data_val = data_val
        self.data_test = data_test
        self.predictors_train = predictors_train
        if self.predictors_train is not None and not isinstance(self.predictors_train, list):
            raise TypeError('`predictors_train` must be a list of ndarrays')
        self.predictors_test = predictors_test
        if self.predictors_test is not None and not isinstance(self.predictors_test, list):
            raise TypeError('`predictors_test` must be a list of ndarrays')
        self.predictors_val = predictors_val
        if self.predictors_val is not None and not isinstance(self.predictors_val, list):
            raise TypeError('`predictors_val` must be a list of ndarrays')
        self.interpolation = interpolation 
        self.epochs = epochs
        self.upsampling = upsampling
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
        self.save_bestmodel = save_bestmodel


    def setup_model(self):
        """
        Setting up the model
        Omitted the static variable code
        """
        n_channels = self.data_train.shape[1] # should be 1
        if self.predictors_train is not None:
            n_channels += len(self.predictors_train) #should be 2

        lr_height = int(self.data_train.shape[-2] / self.scale) #NAM: 40
        lr_width = int(self.data_train.shape[-1] / self.scale) #NAM: 40
        hr_height = int(self.data_train.shape[-2])  #uWFRF: 120
        hr_width = int(self.data_train.shape[-1]) #uWRF: 120


        ### Instantiating the model
        self.model = net_postupsampling(
            backbone_block=self.backbone,
            upsampling=self.upsampling,
            scale=self.scale,
            lr_size=(lr_height, lr_width), # 40 by 40
            n_channels=n_channels, #2
            **self.architecture_params)

        
    def run(self):
        """
        Compiling, training and saving the model
        """
        self.timing = Timing(self.verbose)
        self.setup_model()

        start_time = time.time()

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.L1Loss() if self.loss == 'mae' else nn.MSELoss()
        train_losses = []
        val_losses = []

        self.history = {
            'train_loss': [],
            'val_loss': []
        }

        train_loader = DataLoader(
            DataGenerator(
                array=self.data_train,
                predictors=self.predictors_train,
                backbone=self.backbone,
                upsampling=self.upsampling,
                scale=self.scale
                ),
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=True)

        val_loader = DataLoader(
            DataGenerator(
                array=self.data_val,
                predictors=self.predictors_val,
                backbone=self.backbone,
                upsampling=self.upsampling,
                scale=self.scale
                ),
                batch_size=self.batch_size)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(0, self.epochs):
            self.model.train()
            total_train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Training]", leave=False):
                # x = low resolution input, y = high resolution target
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            self.model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Validation]", leave=False):
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = self.model(x)
                    loss = self.criterion(y_pred, y)
                    total_val_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)

            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)


            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}")
            
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

        self.val_loss = float(val_losses[-1]) if val_losses else float('inf')
        self.training_runtime = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        self.timing.runtime()

        if self.save:
            #print(f"[Saved] Loss curves to {self.save_path}")
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "final_model.pt"))
        
        if self.show_plot:
            # Build full path: save_path directory + fixed name
            learning_curve_path = os.path.join(self.save_path, "learning_curve.png")
            plot_history(
                self.history, 
                title=f"{self.backbone} Training History",
                log_scale=False,
                save_path=learning_curve_path
            )
            