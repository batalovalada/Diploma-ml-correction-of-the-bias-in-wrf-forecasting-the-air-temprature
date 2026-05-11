import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config.data.features_config import features

# ======== eda plots =======================================
def save_corr_matrix_plot(corr, path_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Корреляционная матрица (train)')
    plt.savefig(path_dir + 'corr_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_features_hist(df,  path_dir):
    df[features].hist(bins=20, figsize=(15, 10))
    plt.suptitle('Распределение признаков (train)')
    plt.savefig(path_dir+ 'features_hist.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_spatial_mean_t2m_map(df_wrf, df_era5, path_dir, vmin = 250, vmax=280):
    T2_wrf_mean = df_wrf.groupby(['south_north', 'west_east'])['T2'].mean().unstack()
    T2_era5_mean = df_era5.groupby(['south_north', 'west_east'])['t2m'].mean().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    im0 = axes[0].imshow(T2_wrf_mean.values, cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[0].set_title("WRF mean temperature")
    axes[0].set_xlabel("Longitude index")
    axes[0].set_ylabel("Latitude index")

    im1 = axes[1].imshow(T2_era5_mean.values, cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[1].set_title("ERA5 mean temperature")
    axes[1].set_xlabel("Longitude index")
    axes[1].set_ylabel("Latitude index")

    fig.colorbar(im0, ax=axes, label='Temperature (K)', fraction=0.046, pad=0.04)
    plt.savefig(path_dir + "mean_temperature_map.png")
    plt.close()

def save_seasonal_spatial_mean_t2m_map(df_wrf, df_era5, path_dir):
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(4, 2, figsize=(14, 20), constrained_layout=True)

    for i, season in enumerate(seasons):
        # WRF
        wrf_season = df_wrf[df_wrf["season"] == season]
        T2_wrf_mean = wrf_season.groupby(["south_north", "west_east"])["T2"].mean().unstack()

        im0 = axes[i, 0].imshow(T2_wrf_mean.values, cmap="coolwarm", vmin=250, vmax=300)

        axes[i, 0].set_title(f"WRF mean temperature ({season})")
        axes[i, 0].set_xlabel("Longitude index")
        axes[i, 0].set_ylabel("Latitude index")

        # ERA5
        era5_season = df_era5[df_era5["season"] == season]
        T2_era5_mean = era5_season.groupby(["south_north", "west_east"])["t2m"].mean().unstack()

        im1 = axes[i, 1].imshow(T2_era5_mean.values, cmap="coolwarm", vmin=250, vmax=300)

        axes[i, 1].set_title(f"ERA5 mean temperature ({season})")
        axes[i, 1].set_xlabel("Longitude index")
        axes[i, 1].set_ylabel("Latitude index")

        fig.colorbar(im0, ax=axes[i], label="Temperature (K)", fraction=0.02, pad=0.02)

    plt.savefig(path_dir + "seasonal_mean_temperature_map.png")
    plt.close()

def save_temporal_t2m_dynamics(df_wrf, df_era5, path_dir):
    T2_wrf_time = df_wrf.groupby('time')['T2'].mean()
    T2_era5_time = df_era5.groupby('time')['t2m'].mean()

    plt.figure(figsize=(25, 5))

    plt.plot(T2_wrf_time.index, T2_wrf_time.values, label='WRF')
    plt.plot(T2_era5_time.index, T2_era5_time.values, label='ERA5')

    plt.xlabel("Time step")
    plt.ylabel("Temperature (K)")
    plt.title("Mean temperature temporal evolution")

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(path_dir+ "temperature_timeseries.png")
    plt.close()

def save_spatial_std_map(df_wrf, df_era5, path_dir, vmin_std = 0, vmax_std = 8):
    T2_wrf_std = df_wrf.groupby(['south_north', 'west_east'])['T2'].std().unstack()
    T2_era5_std = df_era5.groupby(['south_north', 'west_east'])['t2m'].std().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    im0 = axes[0].imshow(T2_wrf_std.values, cmap='viridis', vmin=vmin_std, vmax=vmax_std)
    axes[0].set_title("WRF temperature variability (std)")
    axes[0].set_xlabel("Longitude index")
    axes[0].set_ylabel("Latitude index")

    im1 = axes[1].imshow(T2_era5_std.values, cmap='viridis', vmin=vmin_std, vmax=vmax_std)
    axes[1].set_title("ERA5 temperature variability (std)")
    axes[1].set_xlabel("Longitude index")
    axes[1].set_ylabel("Latitude index")

    fig.colorbar(im0, ax=axes, label='Std temperature', fraction=0.046, pad=0.04)
    plt.savefig(path_dir + "temperature_std_map.png")
    plt.close()


def save_seasonal_spatial_std_map(df_wrf, df_era5, path_dir,  vmin_std = 0, vmax_std = 11):
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(4, 2, figsize=(14, 20), constrained_layout=True)

    for i, season in enumerate(seasons):
        # WRF
        wrf_season = df_wrf[df_wrf["season"] == season]
        T2_wrf_std = wrf_season.groupby(["south_north", "west_east"])["T2"].std().unstack()

        im0 = axes[i, 0].imshow(T2_wrf_std.values, cmap="viridis", vmin=vmin_std, vmax=vmax_std)

        axes[i, 0].set_title(f"WRF temperature std ({season})")
        axes[i, 0].set_xlabel("Longitude index")
        axes[i, 0].set_ylabel("Latitude index")

        # ERA5
        era5_season = df_era5[df_era5["season"] == season]
        T2_era5_std = era5_season.groupby(["south_north", "west_east"])["t2m"].std().unstack()

        im1 = axes[i, 1].imshow(T2_era5_std.values, cmap="viridis", vmin=vmin_std, vmax=vmax_std)

        axes[i, 1].set_title(f"ERA5 temperature std ({season})")
        axes[i, 1].set_xlabel("Longitude index")
        axes[i, 1].set_ylabel("Latitude index")

        fig.colorbar(im0, ax=axes[i], label="Temperature std (K)", fraction=0.02, pad=0.02)

    plt.savefig(path_dir + "seasonal_temperature_std_map.png")
    plt.close()

# ========= common plots ====================================
# scatter plot
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

# Spatial RMSE map
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

# correction map with bias error has done by wrf and model
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

# ===================== ML and XGBoost plots ==============================
# feature importance plot
def save_feature_importance_plot(model, lags, path_dir, model_name):
    importances = model.feature_importances_
    feature_names_expanded = []

    for lag in range(lags):
        for f in features:
            feature_names_expanded.append(f"{f}_lag{lag + 1}")

    feature_names_expanded += ["lat", "lon"] #if there are lat lon features

    plt.figure(figsize=(12, 10))
    plt.barh(feature_names_expanded, importances)
    plt.xlabel("Importance")
    plt.title(f"Feature importance ({model_name})")
    plt.tight_layout()
    plt.savefig(path_dir + 'feature_importance.png')
    plt.close()

# residual distribution
def save_radial_distribution_plot(pred, true, mask, path_dir, model_name):
    residuals = pred - true
    residuals_flat = residuals[mask].ravel()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals_flat, bins=50)
    plt.axvline(0, linestyle='--')
    plt.title(f"Residual distribution ({model_name})")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.savefig(path_dir + 'residual_distribution.png')
    plt.close()

# ======================= ConvLSTM plots =================================
# loss plot
def save_loss_plot(train_losses, val_losses, path_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training curve (ConvLSTM)")
    plt.savefig(path_dir + "loss_curve.png")
    plt.close()










