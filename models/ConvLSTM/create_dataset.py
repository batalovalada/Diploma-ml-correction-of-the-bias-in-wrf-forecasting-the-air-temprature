import xarray as xr
import numpy as np

from config.data.features_config import features, spatial_features, temporal_features

# month base
# from config.data.split_month_config import *
# path_selected = '../../data/preprocessed/month/selected/'
# path_t2m = '../../data/preprocessed/month/t2m/'
# path_norm = '../../data/ConvLSTM/month/base/norm_params/'
# path_processed = '../../data/ConvLSTM/month/base/processed/'

# # month spatial
# from config.data.split_month_config import *
# path_selected = '../../data/preprocessed/month/selected/'
# path_t2m = '../../data/preprocessed/month/t2m/'
# path_norm = '../../data/ConvLSTM/month/spatial/norm_params/'
# path_processed = '../../data/ConvLSTM/month/spatial/processed/'
#
# # year base
# from config.data.split_year_config import *
# path_selected = '../../data/preprocessed/year/selected/'
# path_t2m = '../../data/preprocessed/year/t2m/'
# path_norm = '../../data/ConvLSTM/year/base/norm_params/'
# path_processed = '../../data/ConvLSTM/year/base/processed/'

# # year spatial
# from config.data.split_year_config import *
# path_selected = '../../data/preprocessed/year/selected/'
# path_t2m = '../../data/preprocessed/year/t2m/'
# path_norm = '../../data/ConvLSTM/year/spatial/norm_params/'
# path_processed = '../../data/ConvLSTM/year/spatial/processed/'
#
# # year temporal
# from config.data.split_year_config import *
# path_selected = '../../data/preprocessed/year/selected/'
# path_t2m = '../../data/preprocessed/year/t2m/'
# path_norm = '../../data/ConvLSTM/year/temporal/norm_params/'
# path_processed = '../../data/ConvLSTM/year/temporal/processed/'
#
#
# # year spatiotemporal
from config.data.split_year_config import *
path_selected = '../../data/preprocessed/year/selected/'
path_t2m = '../../data/preprocessed/year/t2m/'
path_norm = '../../data/ConvLSTM/year/spatiotemporal/norm_params/'
path_processed = '../../data/ConvLSTM/year/spatiotemporal/processed/'


# ======= parameters ================================================
lookback = 4
horizon = 1

# split data to train, test, val function
def split_data(X_ds, y_ds, y_mask_ds, start, end):
    return (
        X_ds.sel(time=slice(start, end)),
        y_ds.sel(time=slice(start, end)),
        y_mask_ds.sel(time=slice(start, end)),
    )

# normalize function
def normalize(data, mean, std):
    return (data - mean) / std

def normalize_split_data(train_data, val_data, test_data, mean, std):
    return (
        normalize(train_data, mean, std),
        normalize(val_data, mean, std),
        normalize(test_data, mean, std)
    )

# ============== create norm_params function =============
def create_sequences(X, y, mask, lookback, horizon):
    X_seq, y_seq, mask_seq, time_seq = [], [], [], []

    for i in range(len(X.time) - lookback - horizon + 1):
        X_seq.append(X.isel(time=slice(i, i+lookback)).values)

        y_t = i+lookback+horizon-1
        y_seq.append(y.isel(time=y_t).values)
        mask_seq.append(mask.isel(time=y_t).values)
        time_seq.append(X.time.values[y_t])
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(mask_seq, dtype=bool),
        np.array(time_seq, dtype='datetime64[ns]'),
    )

# =============================== load data ============================
ds_wrf = xr.open_dataset(path_selected+'ds_selected_wrf.nc')
ds_era5 = xr.open_dataset(path_selected+'ds_selected_era5.nc')

t2_wrf_ds = xr.open_dataset(path_t2m+'t2_wrf_test.nc')
t2_era5_ds = xr.open_dataset(path_t2m+'t2_era5_test.nc')

# to choose, what dataset do you need:
# base
# ds_wrf = ds_wrf.drop_vars(spatial_features+temporal_features)
# spatial
# ds_wrf = ds_wrf.drop_vars(temporal_features)
# # temporal
# ds_wrf = ds_wrf.drop_vars(spatial_features)
# spatiotemporal - nothing

# creating X, y with train, val, test ======================================
X_ds = ds_wrf
y_ds = ds_era5['t2m'] - ds_wrf['T2']

y_mask = (~np.isnan(y_ds)) # -> loss
y_ds = y_ds.fillna(0) # y_ds has nan

X_train, y_train, y_mask_train = split_data(X_ds, y_ds, y_mask, train_start, train_end)
X_val, y_val, y_mask_val = split_data(X_ds, y_ds, y_mask, val_start, val_end)
X_test, y_test, y_mask_test = split_data(X_ds, y_ds, y_mask, test_start, test_end)

# T2 wrf, era5 to ndarray for check model result ============================
offset = lookback + horizon - 1
t2_wrf= t2_wrf_ds.isel(time=slice(offset, None))['T2'].values
t2_era5 = t2_era5_ds.isel(time=slice(offset, None))['t2m'].values

# ========================== normalize =======================================
X_train = X_train.to_array().transpose("time","variable","south_north","west_east")
X_val = X_val.to_array().transpose("time","variable","south_north","west_east")
X_test = X_test.to_array().transpose("time","variable","south_north","west_east")

X_mean = X_train.sel(variable=features).mean(dim=['time', 'south_north', 'west_east'])
X_std = X_train.sel(variable=features).std(dim=['time', 'south_north', 'west_east'])

y_mean = y_train.mean()
y_std = y_train.std()

X_train.loc[dict(variable=features)] = normalize(X_train.sel(variable=features), X_mean, X_std)
X_val.loc[dict(variable=features)] = normalize(X_val.sel(variable=features), X_mean, X_std)
X_test.loc[dict(variable=features)] = normalize(X_test.sel(variable=features), X_mean, X_std)

# normalized separated spatial features
X_spatial_mean = X_train.sel(variable=spatial_features).mean(dim=['time', 'south_north', 'west_east'])
X_spatial_std = X_train.sel(variable=spatial_features).std(dim=['time', 'south_north', 'west_east'])

X_train.loc[dict(variable=spatial_features)] = normalize(X_train.sel(variable=spatial_features), X_spatial_mean, X_spatial_std)
X_val.loc[dict(variable=spatial_features)] = normalize(X_val.sel(variable=spatial_features), X_spatial_mean, X_spatial_std)
X_test.loc[dict(variable=spatial_features)] = normalize(X_test.sel(variable=spatial_features), X_spatial_mean, X_spatial_std)

# target normalized
y_train, y_val, y_test = normalize_split_data(y_train, y_val, y_test, y_mean, y_std)

# ================================= create sequences  =================================

X_train_seq, y_train_seq, mask_train_seq, time_train_seq = create_sequences(X_train, y_train, y_mask_train, lookback, horizon)
X_val_seq, y_val_seq, mask_val_seq, time_val_seq  = create_sequences(X_val, y_val, y_mask_val, lookback, horizon)
X_test_seq, y_test_seq, mask_test_seq, time_test_seq  = create_sequences(X_test, y_test, y_mask_test, lookback, horizon)

# ========================= save normalize parameters ====================
norm_params_X = {
    'mean': X_mean,
    'std': X_std
}

norm_params_y = {
    'mean': y_mean,
    'std': y_std
}
np.save(path_norm+'norm_params_X.npy', norm_params_X)
np.save(path_norm+'norm_params_y.npy', norm_params_y)

# ========================= save preprocess data ====================
np.savez(path_processed+'train.npz', X=X_train_seq, y=y_train_seq, mask=mask_train_seq, time=time_train_seq)
np.savez(path_processed+'val.npz', X=X_val_seq, y=y_val_seq, mask=mask_val_seq, time=time_val_seq)
np.savez(path_processed+'test.npz', X=X_test_seq, y=y_test_seq, mask=mask_test_seq, time=time_test_seq)

np.save(path_processed+'t2_wrf_test.npy', t2_wrf)
np.save(path_processed+'t2_era5_test.npy', t2_era5)