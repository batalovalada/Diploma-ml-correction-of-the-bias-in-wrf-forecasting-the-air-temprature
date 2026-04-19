import glob
import xarray as xr
import numpy as np

# ========================================= paths ========================
paths_surface = sorted(glob.glob('../../../data/ERA5/surface/ERA5_surface_*.grib'))
path_nodes = '../../../data/lstm/raw/ds_nodes.nc'
path_norm = '../../../data/lstm/norm_params/'
path_processed = '../../../data/lstm/processed/'

# ======= values ================================================
train_start, train_end = '2020-01-01', '2020-01-20'
val_start, val_end = '2020-01-21', '2020-01-25'
test_start, test_end = '2020-01-26', '2020-01-31'

lookback = 4
horizon = 1

# create dataset from .grib files function
def create_dataset_era5(paths, filter_arg):
    ds = xr.concat([xr.open_mfdataset(path, engine="cfgrib", backend_kwargs=filter_arg) for path in paths], dim='time')
    return ds

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
    X_seq, y_seq, mask_seq = [], [], []

    for i in range(len(X.time) - lookback - horizon + 1):
        X_seq.append(X.isel(time=slice(i, i+lookback)).values)
        y_seq.append(y.isel(time=i+lookback+horizon-1).values)
        mask_seq.append(mask.isel(time=i + lookback + horizon - 1).values)
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(mask_seq, dtype=np.float32)
    )


# =============================== download data ============================
ds_era5 = create_dataset_era5(paths_surface, {"filter_by_keys": {"typeOfLevel": "surface"}})
ds_wrf = xr.open_dataset(path_nodes)

# time intersect wrf (era5 has 6h step, wrf output has ~1h step )=============
ds_wrf['time'] = ds_wrf.indexes['time'].floor('h') # wrf and era5 has ms difference
ds_era5['time'] = ds_era5.indexes['time'].floor('h')

common_time = np.intersect1d(ds_wrf.time, ds_era5.time)
ds_wrf = ds_wrf.sel(time=common_time)
ds_era5 = ds_era5.sel(time=common_time)

# selected nodes era5 interpolation =====================================
# get nodes coordinates (XLAT, XLONG)
lats = ds_wrf.XLAT.isel(time=0).reset_coords(drop=True)
lons = ds_wrf.XLONG.isel(time=0).reset_coords(drop=True)

ds_wrf = ds_wrf.drop_vars(['XLAT', 'XLONG'])
ds_era5_interp = ds_era5.sel(time=common_time).interp(latitude=lats, longitude=lons)

# creating X, y with train, val, test ======================================
X_ds = ds_wrf
y_ds = ds_era5_interp['t2m'] - ds_wrf['T2']

y_mask = (~np.isnan(y_ds)).astype(np.uint8) # -> loss
y_ds = y_ds.fillna(0) # y_ds has nan

X_train, y_train, y_mask_train = split_data(X_ds, y_ds, y_mask, train_start, train_end)
X_val, y_val, y_mask_val = split_data(X_ds, y_ds, y_mask, val_start, val_end)
X_test, y_test, y_mask_test = split_data(X_ds, y_ds, y_mask, test_start, test_end)
# save X_test[T2] (WRF test data), ERA5[t2m] -> model.py
X_test['T2'].to_netcdf(path_processed+ "t2_wrf_test.nc")
ds_era5_interp['t2m'].sel(time=slice(test_start, test_end)).to_netcdf(path_processed+ "t2_era5_test.nc")

# ========================== normalize =======================================
X_train = X_train.to_array().transpose("time","variable","south_north","west_east")
X_val = X_val.to_array().transpose("time","variable","south_north","west_east")
X_test = X_test.to_array().transpose("time","variable","south_north","west_east")

X_mean = X_train.mean(dim=['time', 'south_north', 'west_east'])
X_std = X_train.std(dim=['time', 'south_north', 'west_east'])

y_mean = y_train.mean()
y_std = y_train.std()

X_train, X_val, X_test = normalize_split_data(X_train, X_val, X_test, X_mean, X_std)
y_train, y_val, y_test = normalize_split_data(y_train, y_val, y_test, y_mean, y_std)

# ================================= create sequences  =================================

X_train_seq, y_train_seq, mask_train_seq = create_sequences(X_train, y_train, y_mask_train, lookback, horizon)
X_val_seq, y_val_seq, mask_val_seq  = create_sequences(X_val, y_val, y_mask_val, lookback, horizon)
X_test_seq, y_test_seq, mask_test_seq  = create_sequences(X_test, y_test, y_mask_test, lookback, horizon)

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
np.savez(path_processed+'train.npz', X=X_train_seq, y=y_train_seq, mask=mask_train_seq)
np.savez(path_processed+'val.npz', X=X_val_seq, y=y_val_seq, mask=mask_val_seq)
np.savez(path_processed+'test.npz', X=X_test_seq, y=y_test_seq, mask=mask_test_seq)