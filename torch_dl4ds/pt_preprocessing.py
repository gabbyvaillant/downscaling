import numpy as np
import torch
import xarray as xr

class TorchMinMaxScaler:
    def __init__(self, value_range=(0, 1), axis=None, fillnanto=-1):
        self.value_range = value_range
        self.axis = axis
        self.fillnanto = fillnanto

    def fit(self, X):
        if isinstance(X, xr.DataArray):
            X = X.values

        if X.size == 0:
            raise ValueError("Input to fit() is empty")

        X = np.nan_to_num(X, nan=self.fillnanto)

        data_min = np.nanmin(X, axis=self.axis, keepdims=True)
        data_max = np.nanmax(X, axis=self.axis, keepdims=True)
        data_range = data_max - data_min

        self.scale_ = (self.value_range[1] - self.value_range[0]) / np.where(data_range == 0, 1, data_range)
        self.min_ = self.value_range[0] - data_min * self.scale_
        return self

    def transform(self, X):
        if isinstance(X, xr.DataArray):
            X = X.values

        if X.size == 0:
            raise ValueError("Input to transform() is empty")

        X = np.nan_to_num(X, nan=self.fillnanto)

        if X.ndim == 0:
            X = np.expand_dims(X, axis=0)

        X = X * self.scale_ + self.min_
        return torch.tensor(X, dtype=torch.float32)

    def inverse_transform(self, X):
        X = X.detach().cpu().numpy()

        if X.ndim == 0:
            X = np.expand_dims(X, axis=0)

        return (X - self.min_) / self.scale_


class TorchStandardScaler:
    def __init__(self, with_mean=True, with_std=True, axis=None, fillnanto=0):
        self.with_mean = with_mean
        self.with_std = with_std
        self.axis = axis
        self.fillnanto = fillnanto

    def fit(self, X):
        if isinstance(X, xr.DataArray):
            X = X.values

        if X.size == 0:
            raise ValueError("Input to fit() is empty")

        X = np.nan_to_num(X, nan=self.fillnanto)

        if self.with_mean:
            self.mean_ = np.nanmean(X, axis=self.axis, keepdims=True)
        if self.with_std:
            self.std_ = np.nanstd(X, axis=self.axis, keepdims=True)
            self.std_ = np.where(self.std_ == 0, 1, self.std_)

        return self

    def transform(self, X):
        if isinstance(X, xr.DataArray):
            X = X.values

        if X.size == 0:
            raise ValueError("Input to transform() is empty")

        X = np.nan_to_num(X, nan=self.fillnanto)

        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.std_

        if X.ndim == 0:
            X = np.expand_dims(X, axis=0)

        return torch.tensor(X, dtype=torch.float32)

    def inverse_transform(self, X):
        X = X.detach().cpu().numpy()

        if X.ndim == 0:
            X = np.expand_dims(X, axis=0)

        if self.with_std:
            X *= self.std_
        if self.with_mean:
            X += self.mean_

        return X
