from sklearn.ensemble import RandomForestRegressor
import joblib

from config.data.features_config import features
from config.data.split_config import *
from config.models_params.rf import *
from metrics.metrics import *
from visualization.save_metrics import save_metrics
from visualization.save_plots import *

# ============ paths =============
path_processed = '../../data/RF/processed/'
path_params = '../../data/RF/params/'
path_results = '../../reports/results/RF/'

# ========== parameters ==========
H = 5 # count of points in height
W = 5 # count of points in width

# ======== function ==============
def load_npz(path):
    data = np.load(path)
    return data['X'], data['y'], data['mask']

def restore_masked_grid(mask, samples, time, H, W):
    bias_pred = np.full(mask.shape, np.nan)
    bias_pred[mask] = samples
    bias_pred= bias_pred.reshape(time, H, W)
    return bias_pred


# ============ load data =========
X_train, y_train, mask_train = load_npz(path_processed + 'train.npz')
X_val, y_val, mask_val = load_npz(path_processed + 'val.npz')
X_test, y_test, mask_test = load_npz(path_processed + 'test.npz')

T2_wrf_test = np.load(path_processed+'t2_wrf_test.npy')
T2_era5_test = np.load(path_processed+'t2_era5_test.npy')
T2_mask_test = np.load(path_processed+'t2_mask_test.npy')

time_lagged = np.load(path_params+'time_lagged.npy')
time_test_mask = np.load(path_processed+'rf_time_mask.npy')
# ============= model ============
RF = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH_SIZE,
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
    max_features=MAX_FEATURES,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    min_samples_split=MIN_SAMPLES_SPLIT,
)

# =========== train ================
RF.fit(X_train, y_train)

# ======== validation to correct models_params ========
val_pred = RF.predict(X_val)

# =========== test =========================
test_pred = RF.predict(X_test)

# ======== restore val data (to select model's models_params on validation) ==========
val_mask = ((time_lagged >= val_start_dt) & (time_lagged <= val_end_dt))
time_val = len(time_lagged[val_mask])

val_pred_restored = restore_masked_grid(mask_val, val_pred, time_val, H, W)
y_val_restored = restore_masked_grid(mask_val, y_val, time_val, H, W)
mask_val_reshaped = mask_val.reshape(time_val, H, W)

# ============= Val Metrics ==========================
rmse = masked_rmse(val_pred_restored, y_val_restored, mask_val_reshaped)
mae = masked_mae(val_pred_restored, y_val_restored, mask_val_reshaped)
bias = masked_bias(val_pred_restored, y_val_restored, mask_val_reshaped)
corr = masked_corr(val_pred_restored, y_val_restored, mask_val_reshaped)

print("\n\nVal metrics:")
print(f"RMSE: {rmse}")
print(f"MAE:  {mae}")
print(f"Bias: {bias}")
print(f"Corr: {corr}")

with open(path_results+'validation/'+ f"metrics_1.txt", "w") as f:
    f.write("RF params\n\n")
    f.write(f"lags: {lags}\n")
    f.write(f"N_ESTIMATORS: {N_ESTIMATORS}\n")
    f.write(f"MAX_DEPTH_SIZE: {MAX_DEPTH_SIZE}\n")
    f.write(f"N_JOBS: {N_JOBS}\n")
    f.write(f"RANDOM_STATE: {RANDOM_STATE}\n")
    f.write(f"MAX_FEATURES: {MAX_FEATURES}\n")
    f.write(f"MIN_SAMPLES_LEAF: {MIN_SAMPLES_LEAF}\n")
    f.write(f"MIN_SAMPLES_SPLIT: {MIN_SAMPLES_SPLIT}\n\n\n")
    f.write("Validation Metrics\n\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"MAE: {mae}\n")
    f.write(f"Bias: {bias}\n")
    f.write(f"Corr: {corr}\n\n")



# #============= restore test data =============================
test_mask = ((time_lagged >= test_start_dt) & (time_lagged <= test_end_dt))
T_test= len(time_lagged[test_mask])

test_pred_restored = restore_masked_grid(mask_test, test_pred, T_test, H, W)
y_test_restored = restore_masked_grid(mask_test, y_test, T_test, H, W)
mask_test_reshaped = mask_test.reshape(T_test, H, W)

test_pred_restored = test_pred_restored[time_test_mask]
y_test_restored = y_test_restored[time_test_mask]
mask_test_reshaped = mask_test_reshaped[time_test_mask]

T2_corrected = T2_wrf_test + test_pred_restored

# ============= Test Metrics ==========================
rmse = masked_rmse(test_pred_restored, y_test_restored, mask_test_reshaped)
mae = masked_mae(test_pred_restored, y_test_restored, mask_test_reshaped)
bias = masked_bias(test_pred_restored, y_test_restored, mask_test_reshaped)
corr = masked_corr(test_pred_restored, y_test_restored, mask_test_reshaped)

print("\nTest metrics:")
print(f"RMSE: {rmse}")
print(f"MAE:  {mae}")
print(f"Bias: {bias}")
print(f"Corr: {corr}")

#rmse metric to compare WRF and RF
rmse_wrf = masked_rmse(T2_wrf_test, T2_era5_test, T2_mask_test)
rmse_corr = masked_rmse(T2_corrected, T2_era5_test, T2_mask_test)

print(f"WRF RMSE: {rmse_wrf}")
print(f"Corrected RMSE: {rmse_corr}")

# vizualization ========================================
# feature importance plot
importances = RF.feature_importances_
feature_names_expanded = []

for lag in range(lags):
    for f in features:
        feature_names_expanded.append(f"{f}_lag{lag+1}")

feature_names_expanded += ["lat", "lon"]

plt.figure(figsize=(12,10))
plt.barh(feature_names_expanded, importances)
plt.xlabel("Importance")
plt.title("Feature importance (RF)")
plt.tight_layout()
plt.savefig(path_results+'feature_importance.png')
plt.close()

# residual distribution
residuals = test_pred_restored - y_test_restored
residuals_flat = residuals[mask_test_reshaped].ravel()

plt.figure(figsize=(10,6))
plt.hist(residuals_flat, bins=50)
plt.axvline(0, linestyle='--')
plt.title("Residual distribution (RF)")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.savefig(path_results+'residual_distribution.png')
plt.close()

# scatter plot
save_scatter_plot(test_pred_restored, y_test_restored, mask_test_reshaped, path_results, 'RF')

# Spatial RMSE map
save_rmse_map(test_pred_restored, y_test_restored, mask_test_reshaped, path_results, 'RF')

# Random Forest visualization
save_corrected_map(T2_wrf_test, T2_era5_test, T2_corrected, path_results, 'RF')

# save results and model ============================================
joblib.dump(RF, path_results+"rf_model.pkl")

# save results
np.savez(
    path_results+"results_rf.npz",
    y_pred=test_pred_restored,
    y_true=y_test_restored,
    mask=mask_test_reshaped,
)

# save metrics
save_metrics(path_results, rmse, mae, bias, corr, rmse_wrf, rmse_corr)