import numpy as np
from scipy.stats import pearsonr, spearmanr

def compute_basic_metrics(y_true, y_pred, mask=None):
    """
    Compute MAE, RMSE, Pearson and Spearman correlation across time.

    Parameters
    ----------
    y_true : np.ndarray, shape (time, height, width)
    y_pred : np.ndarray, shape (time, height, width)
    mask : np.ndarray or None, shape (height, width)
        Optional spatial mask to apply before computing metrics.

    Returns
    -------
    dict of str -> float
        Dictionary with average metrics.
    """
    if mask is not None:
        y_true = y_true * mask
        y_pred = y_pred * mask

    # Reshape to (time, space)
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    # Filter out invalid (zero-only) columns
    valid = ~np.all(y_true_flat == 0, axis=0)
    y_true_flat = y_true_flat[:, valid]
    y_pred_flat = y_pred_flat[:, valid]

    # Compute metrics per timestep
    maes = [np.mean(np.abs(y_true_flat[t] - y_pred_flat[t])) for t in range(y_true.shape[0])]
    rmses = [np.sqrt(np.mean((y_true_flat[t] - y_pred_flat[t]) ** 2)) for t in range(y_true.shape[0])]
    pearsons = [pearsonr(y_true_flat[t], y_pred_flat[t])[0] for t in range(y_true.shape[0])]
    spearmans = [spearmanr(y_true_flat[t], y_pred_flat[t])[0] for t in range(y_true.shape[0])]

    return {
        'MAE': np.mean(maes),
        'RMSE': np.mean(rmses),
        'Pearson': np.nanmean(pearsons),
        'Spearman': np.nanmean(spearmans)
    }