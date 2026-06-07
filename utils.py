import json
import numpy as np


def calculate_metrics(y_true, y_pred):
    """
    Compute MAE, RMSE, NMAE, NRMSE, R2.

    Args:
        y_true: numpy array of ground-truth values
        y_pred: numpy array of predicted values
    Returns:
        dict with keys: mae, rmse, nmae, nrmse, r2
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    y_range = y_true.max() - y_true.min()
    if y_range > 0:
        nmae = mae / y_range
        nrmse = rmse / y_range
    else:
        nmae = 0.0
        nrmse = 0.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'nmae': float(nmae),
        'nrmse': float(nrmse),
        'r2': float(r2)
    }

