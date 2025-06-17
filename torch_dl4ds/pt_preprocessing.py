import numpy as np
import torch
import xarray as xr

class TorchMinMaxScaler:
    def __init__(self, value_range=(0, 1), axis=None, fillnanto=-1):
        self.value_range = value_range
        self.axis = axis
        self.fillnanto = fillnanto

    def fit(self, X):
        X = np.squeeze(X)

        if np.any(np.isnan(X)):
            self.nan_mask = np.isnan(X)

        if isinstance(X, xr.DataArray):
            X = X.values

        data_min = np.nanmin(X, axis=self.axis, keepdims=True)
        data_max = np.nanmax(X, axis=self.axis, keepdims=True)
        data_range = data_max - data_min

        self.scale_ = (self.value_range[1] - self.value_range[0]) / np.where(data_range == 0, 1, data_range)
        self.min_ = self.value_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        X = np.squeeze(X.copy())

        X = X * self.scale_ + self.min_
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=self.fillnanto)
        return torch.tensor(X, dtype=torch.float32)

    def inverse_transform(self, X):
        X = X.detach().cpu().numpy()
        X = np.squeeze(X.copy())

        X = (X - self.min_) / self.scale_
        if hasattr(self, 'nan_mask'):
            X[self.nan_mask] = np.nan
        return X


class TorchStandardScaler:
    def __init__(self, with_mean=True, with_std=True, axis=None, fillnanto=0):
        self.with_mean = with_mean
        self.with_std = with_std
        self.axis = axis
        self.fillnanto = fillnanto

    def fit(self, X):
        X = np.squeeze(X)

        if np.any(np.isnan(X)):
            self.nan_mask = np.isnan(X)

        if isinstance(X, xr.DataArray):
            X = X.values

        if self.with_mean:
            self.mean_ = np.nanmean(X, axis=self.axis, keepdims=True)
        if self.with_std:
            self.std_ = np.nanstd(X, axis=self.axis, keepdims=True)
        return self

    def transform(self, X):
        X = np.squeeze(X.copy())
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.std_
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=self.fillnanto)
        return torch.tensor(X, dtype=torch.float32)

    def inverse_transform(self, X):
        X = X.detach().cpu().numpy()
        X = np.squeeze(X.copy())

        if self.with_std:
            X *= self.std_
        if self.with_mean:
            X += self.mean_

        if hasattr(self, 'nan_mask'):
            X[self.nan_mask] = np.nan
        return X
