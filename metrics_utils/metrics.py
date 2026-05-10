import numpy as np

def masked_rmse(pred, target, mask):
    return np.sqrt(np.mean((pred[mask] - target[mask]) ** 2))

def masked_mae(pred, target, mask):
    return np.mean(np.abs(pred[mask] - target[mask]))

def masked_bias(pred, target, mask):
    return np.mean(pred[mask] - target[mask])

def masked_corr(pred, target, mask):
    return np.corrcoef(pred[mask], target[mask])[0, 1]


def define_and_save_metrics(pred, true, mask, T2_wrf, T2_corrected, T2_era5, path_dir):
    rmse = masked_rmse(pred, true, mask)
    mae = masked_mae(pred, true, mask)
    bias = masked_bias(pred, true, mask)
    corr = masked_corr(pred, true, mask)

    print("\nTest metrics:")
    print(f"RMSE: {rmse}")
    print(f"MAE:  {mae}")
    print(f"Bias: {bias}")
    print(f"Corr: {corr}")

    # rmse metric to compare WRF and ML
    rmse_wrf = masked_rmse(T2_wrf, T2_era5, mask)
    rmse_corr = masked_rmse(T2_corrected, T2_era5, mask)

    print(f"WRF RMSE: {rmse_wrf}")
    print(f"Corrected RMSE: {rmse_corr}")

    with open(path_dir + "metrics.txt", "w") as f:
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"Bias: {bias}\n")
        f.write(f"Corr: {corr}\n\n")
        f.write(f"Compare WRF and Corrected\n")
        f.write(f"WRF RMSE: {rmse_wrf}\n")
        f.write(f"Corrected RMSE: {rmse_corr}\n")