import joblib
import optuna
from sklearn.metrics import mean_squared_error

from config.hyperparameters.tree_models import lags, H, W
from config.data.split_month_config import *
from metrics_utils.metrics import *
from visualization.save_plots import *

# depend on model: XGB, RF
from XGB.model import return_params, build_model, train_model, save_best_params
#from RF.model import return_params, build_model, train_model, save_best_params

# ============ paths =============
path_processed = '../../data/tree_models/month/latlon/processed/'
path_params = '../../data/tree_models/month/latlon/params/'

# depend on model: XGB, RF
path_results = '../../reports/models/XGB/month/latlon/'
#path_results = '../../reports/models/RF/month/latlon/'
name_model = 'XGBoost'
#name_model = 'RF'

# ======== function ==============
def load_npz(path):
    data = np.load(path)
    return data['X'], data['y'], data['mask']

def restore_masked_grid(mask, samples, time, H, W):
    bias_pred = np.full(mask.shape, np.nan)
    bias_pred[mask] = samples
    bias_pred= bias_pred.reshape(time, H, W)
    return bias_pred

def objective(trial):
    params = return_params(trial)
    model = build_model(params)
    train_model(model, X_train, y_train, X_val, y_val)
    val_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, val_pred))

# ============ load data =========
X_train, y_train, mask_train = load_npz(path_processed + 'train.npz')
X_val, y_val, mask_val = load_npz(path_processed + 'val.npz')
X_test, y_test, mask_test = load_npz(path_processed + 'test.npz')

T2_wrf_test = np.load(path_processed+'t2_wrf_test.npy')
T2_era5_test = np.load(path_processed+'t2_era5_test.npy')

time_lagged = np.load(path_params+'time_lagged.npy')
time_test_mask = np.load(path_processed+'ml_time_mask.npy')

# ======== select params with Optuna ========
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
best_params = study.best_params

print("Best params:", best_params)
print("Best value:", study.best_value)

# =========== train ================
best_model = build_model(best_params)
train_model(best_model, X_train, y_train, X_val, y_val)

# =========== test =========================
test_pred = best_model.predict(X_test)

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

# save and print metrics ==========================
define_and_save_metrics(test_pred_restored, y_test_restored, mask_test_reshaped, T2_wrf_test, T2_corrected, T2_era5_test, path_results)

# vizualization ========================================
save_feature_importance_plot(best_model, lags, path_results, name_model)
save_radial_distribution_plot(test_pred_restored,y_test_restored, mask_test_reshaped, path_results, name_model)
save_scatter_plot(test_pred_restored, y_test_restored, mask_test_reshaped, path_results, name_model)
save_rmse_map(test_pred_restored, y_test_restored, mask_test_reshaped, path_results, name_model)
save_corrected_map(T2_wrf_test, T2_era5_test, T2_corrected, path_results, name_model)

# save model ============================================
joblib.dump(best_model, path_results+f"{name_model}_model.pkl")

# save results
np.savez(
    path_results+f"results_{name_model}.npz",
    y_pred=test_pred_restored,
    y_true=y_test_restored,
    mask=mask_test_reshaped,
)

# save train params
save_best_params(best_params, path_results)