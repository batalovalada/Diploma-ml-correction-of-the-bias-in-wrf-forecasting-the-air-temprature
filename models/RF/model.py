from sklearn.ensemble import RandomForestRegressor
import joblib
import optuna

from config.data.features_config import features
from config.data.split_config import *
from config.models_params.rf import *
from metrics.metrics import *
from visualization.save_metrics import save_metrics
from visualization.save_plots import *

# ============ paths =============
path_processed = '../../data/RF/processed/'
path_params = '../../data/RF/params/'
path_results = '../../reports/results/RF/opt/'

# ========== parameters ==========
H = 5 # count of points in height
W = 5 # count of points in width

# ======== function ==============
def load_npz(path):
    data = np.load(path)
    return data['X'], data['y'], data['mask']

def mse(pred, target):
    return np.mean((pred - target)**2)

def restore_masked_grid(mask, samples, time, H, W):
    bias_pred = np.full(mask.shape, np.nan)
    bias_pred[mask] = samples
    bias_pred= bias_pred.reshape(time, H, W)
    return bias_pred

def objective(trial):

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 35),
        'max_features': trial.suggest_categorical('max_features', ["sqrt", "log2", 0.5, 0.7]),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1,8),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
    }

    model = RandomForestRegressor(
        **params,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)

    return mse(y_val, val_pred)

# ============ load data =========
X_train, y_train, mask_train = load_npz(path_processed + 'train.npz')
X_val, y_val, mask_val = load_npz(path_processed + 'val.npz')
X_test, y_test, mask_test = load_npz(path_processed + 'test.npz')

T2_wrf_test = np.load(path_processed+'t2_wrf_test.npy')
T2_era5_test = np.load(path_processed+'t2_era5_test.npy')
T2_mask_test = np.load(path_processed+'t2_mask_test.npy')

time_lagged = np.load(path_params+'time_lagged.npy')
time_test_mask = np.load(path_processed+'rf_time_mask.npy')

# ======== select params with Optuna ========
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

best_params = study.best_params
print(best_params)

# =========== train ================
RF = RandomForestRegressor(
    **best_params,
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE
)

RF.fit(X_train, y_train)

# =========== test =========================
test_pred = RF.predict(X_test)

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