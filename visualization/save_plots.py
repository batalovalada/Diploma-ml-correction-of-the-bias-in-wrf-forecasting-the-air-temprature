import matplotlib.pyplot as plt
import numpy as np

# scatter plot ----------------------------
def save_scatter_plot(pred, true, mask, path_dir, model_name):
    mask_flat = mask.flatten().astype(bool)

    x = true.flatten()[mask_flat]
    y = pred.flatten()[mask_flat]

    lims = [min(x.min(), y.min()), max(x.max(), y.max())]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.3)
    plt.plot(lims, lims, 'r--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} Prediction vs Truth")
    plt.savefig(path_dir+"compare.png")
    plt.close()

# Spatial RMSE map -------------------------
def save_rmse_map(pred, true, mask, path_dir, model_name, vmin=0, vmax=8):
    diff = (pred - true) ** 2
    numerator = np.sum(diff * mask, axis=0)
    denominator = np.sum(mask, axis=0)
    denominator = np.where(denominator == 0, np.nan, denominator)

    rmse_map = np.sqrt(numerator / denominator)
    plt.figure(figsize=(10, 6))
    plt.imshow(rmse_map, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f"{model_name} RMSE map")
    plt.savefig(path_dir+"rmse_map.png")
    plt.close()

# correction map with bias error has done by wrf and model ============================
def corrected_map_row(ax_raw, fig, T2_wrf, T2_era5, T2_corrected, error_wrf, error_corr, model_name, vmin=235, vmax=290, vmin_err=-10, vmax_err=10):
    # WRF
    im_t2 = ax_raw[0].imshow(T2_wrf, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax_raw[0].set_title("WRF")

    # Corrected
    ax_raw[1].imshow(T2_corrected, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax_raw[1].set_title(f"{model_name} corrected")

    # ERA5
    ax_raw[2].imshow(T2_era5, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax_raw[2].set_title("ERA5 (truth)")

    # Error WRF
    im_err = ax_raw[3].imshow(error_wrf, cmap="bwr", vmin=vmin_err, vmax=vmax_err)
    ax_raw[3].set_title("WRF error (WRF - ERA5)")

    # Model Error
    ax_raw[4].imshow(error_corr, cmap="bwr", vmin=vmin_err, vmax=vmax_err)
    ax_raw[4].set_title(f"{model_name} error (Corrected - ERA5)")

    fig.colorbar(im_t2, ax=ax_raw[2], fraction=0.046, pad=0.04)
    fig.colorbar(im_err, ax=ax_raw[4], fraction=0.046, pad=0.04)


def save_corrected_map(T2_wrf, T2_era5, T2_corrected, path_dir, model_name):
    count_times = T2_wrf.shape[0]
    time_arr = np.linspace(0, count_times - 1, 4, dtype=int) #select 4 time moments
    # add time moment with max wrf error
    error = np.nanmean(np.abs(T2_wrf - T2_era5), axis=(1, 2))
    idx_max_error = np.argmax(error)
    time_arr= list(set(time_arr.tolist() + [idx_max_error]))
    time_arr.sort()
    count_times = len(time_arr)

    error_wrf = T2_wrf - T2_era5
    error_corr = T2_corrected - T2_era5

    # last row is mean values on grid
    with np.errstate(invalid='ignore'):
        mean_wrf = np.nanmean(T2_wrf, axis=0)
        mean_era5 = np.nanmean(T2_era5, axis=0)
        mean_corr = np.nanmean(T2_corrected, axis=0)

        mean_error_wrf = np.nanmean(error_wrf, axis=0)
        mean_error_corr = np.nanmean(error_corr, axis=0)

    fig, axes = plt.subplots(count_times + 1, 5, figsize=(23, 4 * (count_times + 1)))

    # subplots in each time moment from time_arr
    for i, t in enumerate(time_arr):
        corrected_map_row(axes[i], fig, T2_wrf[t], T2_era5[t], T2_corrected[t], error_wrf[t], error_corr[t], model_name)
        axes[i][0].set_ylabel(f"t = {t}")

    # Mean by time
    corrected_map_row(axes[-1], fig, mean_wrf, mean_era5, mean_corr, mean_error_wrf, mean_error_corr, model_name)
    axes[-1][0].set_ylabel("mean")

    plt.tight_layout()
    plt.savefig(path_dir+"results_map.png")
    plt.close()
