import xarray as xr
import numpy as np
import glob

# ========= paths =======================
path_full = '../data/base_processed/full/'
path_selected = '../data/base_processed/selected/'
path_wrf = '../data/WRF_output/wrfout_d01_*'
paths_surface = sorted(glob.glob('../data/ERA5/surface/ERA5_surface_*.grib'))

# create dataset from .grib files function
def create_dataset_era5(paths, filter_arg):
    ds = xr.concat([xr.open_mfdataset(path, engine="cfgrib", backend_kwargs=filter_arg) for path in paths], dim='time')
    return ds


# =========== parameters =================
features = ['T2', 'PSFC', 'U10', 'V10', 'Q2', 'SWDOWN', 'GLW', 'LH', 'HFX', 'PBLH', 'UST']

sn = [0, 15, 29, 45, 58]  #25 nodes
we = [0, 12, 23, 35, 46]

# =========== download wrf output and era5 data ==============
ds_wrf = xr.open_mfdataset(
    path_wrf,
    combine='by_coords',
    engine='netcdf4',
    decode_times=True,
    chunks={'Time': 24},
)

ds_era5 = create_dataset_era5(paths_surface, {"filter_by_keys": {"typeOfLevel": "surface"}})

# ========== preprocessed wrf data: select features, create DataFrame, rename columns ========
# making 'XTIME' -> Coordinates and rename to 'time'
ds_wrf = ds_wrf.swap_dims({'Time': 'XTIME'})
ds_wrf = ds_wrf.rename({'XTIME': 'time'})

ds_wrf = ds_wrf[features]
df_wrf = ds_wrf.to_dataframe()
# save primary df[features] -------------------------------------
df_wrf.to_parquet(path_full + 'df_full_wrf.parquet', index=True) # -> eda.py
ds_wrf.to_netcdf(path_full + "ds_full_wrf.nc")

# ========== select nodes sn x we, time intersect ==============================
ds_wrf = ds_wrf.isel(south_north=sn, west_east=we)

# era5 has 6h step, wrf output has ~1h step
ds_wrf['time'] = ds_wrf.indexes['time'].floor('h') # wrf and era5 have ms difference
ds_era5['time'] = ds_era5.indexes['time'].floor('h')

common_time = np.intersect1d(ds_wrf.time, ds_era5.time)
ds_wrf = ds_wrf.sel(time=common_time)
ds_era5 = ds_era5.sel(time=common_time)

# selected nodes era5 interpolation
# get nodes coordinates (XLAT, XLONG)
lats = ds_wrf.XLAT.isel(time=0).reset_coords(drop=True)
lons = ds_wrf.XLONG.isel(time=0).reset_coords(drop=True)

ds_wrf = ds_wrf.drop_vars(['XLAT', 'XLONG'])
ds_era5 = ds_era5.sel(time=common_time).interp(latitude=lats, longitude=lons)

# to save selected data in DataFrame
df_wrf = ds_wrf.to_dataframe().reset_index()
df_era5 = ds_era5.to_dataframe().reset_index()
# south_north, west_east -> node_id (using indexes from sn, we lists: sni-wei)
df_wrf['node_id'] = df_wrf['south_north'].astype(str) + '-' + df_wrf['west_east'].astype(str)

# save clean df, ds in selected nodes and intersected time -------------------------------------
df_wrf.to_parquet(path_selected + 'df_selected_wrf.parquet', index=False) # -> eda.py
ds_wrf.to_netcdf(path_selected + "ds_selected_wrf.nc") # -> model preprocess
df_era5.to_parquet(path_selected + 'df_selected_era5.parquet', index=False) # -> model preprocess
ds_era5.to_netcdf(path_selected + 'ds_selected_era5.nc') # -> model preprocess