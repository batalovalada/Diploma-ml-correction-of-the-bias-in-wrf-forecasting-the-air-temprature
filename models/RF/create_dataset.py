import numpy as np
import xarray as xr

from config.data.split_config import *
from config.data.features_config import features
from config.models_params.rf import lags

# ============= paths =======================
path_selected = '../../data/base_processed/selected/'
path_t2m =  '../../data/base_processed/t2m/'
path_processed = '../../data/RF/processed/'
path_params = '../../data/RF/params/'
path_min_time = '../../data/ConvLSTM/base/processed/'

#===========functions =========================
def split_by_mask(X, y, mask, time_lagged):
    return X[mask], y[mask], time_lagged[mask]

def flatten_data(X, y):
    t, f, h, w = X.shape
    X_flat = X.transpose(0, 2, 3, 1).reshape(-1, f)  # (samples, features), samples = time x H x W
    y_flat = y.reshape(-1)  # (time x H x W, )
    return X_flat, y_flat, t, h, w

def add_coords(X, lat_flatten, lon_flatten, time):
    # add coordinate to each time moment
    lat_tiled = np.tile(lat_flatten, time)
    lon_tiled = np.tile(lon_flatten, time)

    return np.concatenate([X, lat_tiled[:, None], lon_tiled[:, None]], axis=1)

def apply_nan_mask(X, y):
    mask = ~np.isnan(y)
    return X[mask], y[mask], mask

# ============ load data ================
ds_wrf = xr.open_dataset(path_selected+'ds_selected_wrf.nc')
ds_era5 = xr.open_dataset(path_selected+'ds_selected_era5.nc')

t2_wrf_test = xr.open_dataset(path_t2m+'t2_wrf_test.nc')
t2_era5_test = xr.open_dataset(path_t2m+'t2_era5_test.nc')

# time interval to intersect time to correct comparison
time_lstm_test = np.load(path_min_time+'test.npz')['time']

# =============== create target =============
target = ds_era5['t2m'] - ds_wrf['T2']

# =============== create time lags ==========
X_list = []
y_list = []

for t in range(lags, len(ds_wrf.time)):
    X_t = []
    for lag in range(lags):
        X_t.append(ds_wrf[features].isel(time=t-lag).to_array().values)

    X_t = np.concatenate(X_t, axis=0) # shape = (features*lags, H, W)
    y_t = target.isel(time=t).values # shape = (H, W)

    X_list.append(X_t)
    y_list.append(y_t)

# add time axis
X = np.stack(X_list)  # (time, features*lags, H, W)
y = np.stack(y_list)  # (time, H, W)

#======================= split ===============================
time_lagged = ds_wrf.time[lags:] # for first lag we start from index=lags

train_mask = ((time_lagged >= train_start_dt) & (time_lagged <= train_end_dt))
val_mask = ((time_lagged >= val_start_dt) & (time_lagged <= val_end_dt))
test_mask = ((time_lagged >= test_start_dt) & (time_lagged <= test_end_dt))

X_train, y_train, time_rf_train = split_by_mask(X, y, train_mask, time_lagged)
X_val, y_val, time_rf_val = split_by_mask(X, y, val_mask, time_lagged)
X_test, y_test, time_rf_test = split_by_mask(X, y, test_mask, time_lagged)

# ============= flatten ====================
X_train, y_train, T_train, H, W = flatten_data(X_train, y_train)
X_val, y_val, T_val, _, _ = flatten_data(X_val, y_val)
X_test, y_test, T_test, _, _ = flatten_data(X_test, y_test)

# ================ add lat, lon as a features =======
# select nodes coordinates
lats = ds_wrf.XLAT.isel(time=0).values # shape = (H, W)
lons = ds_wrf.XLONG.isel(time=0).values

lat_flatten = lats.flatten() # shape = (H*W, )
lon_flatten = lons.flatten()

X_train = add_coords(X_train, lat_flatten, lon_flatten, T_train)
X_val = add_coords(X_val, lat_flatten, lon_flatten, T_val)
X_test = add_coords(X_test, lat_flatten, lon_flatten, T_test)

# ============ delete samples with nan values ========
# era5 has nan values => y has too
X_train, y_train, mask_train = apply_nan_mask(X_train, y_train)
X_val, y_val, mask_val = apply_nan_mask(X_val, y_val)
X_test, y_test, mask_test = apply_nan_mask(X_test, y_test)

# ========= T2 to ndarray, add mask (ERA5 has nan), select min time interval =======================
time_common = np.intersect1d(time_lstm_test, time_rf_test)
time_mask_rf = np.isin(time_rf_test, time_common)
#mask_lstm = np.isin(time_lstm_test, time_common)

t2_wrf_test = t2_wrf_test['T2'].values[time_mask_rf] # -> shape (time, H, W)
t2_era5_test = t2_era5_test['t2m'].values[time_mask_rf] # -> shape (time, H, W)
t2_mask_test = ~np.isnan(t2_era5_test) # shape (time, H, W)

# =========== save data ====================
np.save(path_params+'time_lagged.npy', time_lagged.values)
np.save(path_params+'lat_2d.npy', lats)
np.save(path_params+'lon_2d.npy', lons)

np.save(path_processed+'t2_wrf_test.npy', t2_wrf_test)
np.save(path_processed+'t2_era5_test.npy', t2_era5_test)
np.save(path_processed+'t2_mask_test.npy', t2_mask_test)

# save time mask
np.save(path_processed+'rf_time_mask.npy', time_mask_rf)

np.savez(path_processed+'train.npz', X=X_train, y=y_train, mask=mask_train)
np.savez(path_processed+'val.npz', X=X_val, y=y_val, mask=mask_val)
np.savez(path_processed+'test.npz', X=X_test, y=y_test, mask=mask_test)
