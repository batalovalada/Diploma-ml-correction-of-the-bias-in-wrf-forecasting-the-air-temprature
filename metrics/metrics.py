import numpy as np

def masked_rmse(pred, target, mask):
    return np.sqrt(np.mean((pred[mask] - target[mask]) ** 2))

def masked_mae(pred, target, mask):
    return np.mean(np.abs(pred[mask] - target[mask]))

def masked_bias(pred, target, mask):
    return np.mean(pred[mask] - target[mask])

def masked_corr(pred, target, mask):
    return np.corrcoef(pred[mask], target[mask])[0, 1]