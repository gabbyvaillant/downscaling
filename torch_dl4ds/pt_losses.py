import torch as pt
import torch.nn as nn


mae_loss_fn = nn.L1Loss()
mse_loss_fn = nn.MSELoss()

def mae(y_true, y_pred):
    """
    Mean absolute error, L1 pixel loss
    """
    return mae_loss_fn(y_pred, y_true)

def mse(y_true, y_pred):
    """
    Mean squared error, L2 pixel loss
    """
    return mse_loss_fn(y_pred, y_true)
