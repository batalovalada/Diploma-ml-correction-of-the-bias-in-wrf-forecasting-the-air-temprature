import xarray as xr

# ========= paths =======================
data_dir_path = '../../../data/lstm/raw/'
wrf_path = '../../../data/WRF_output/wrfout_d01_*'

# =========== download wrf output data ==============
ds = xr.open_mfdataset(
    wrf_path,
    combine='by_coords',
    engine='netcdf4',
    decode_times=True,
    chunks={'Time': 24},
)

# =========== parameters =================
features = ['T2', 'PSFC', 'U10', 'V10', 'Q2', 'SWDOWN', 'GLW', 'LH', 'HFX', 'PBLH', 'UST']

sn = [0, 15, 29, 45, 58]  #25 nodes
we = [0, 12, 23, 35, 46]

train_start, train_end = '2020-01-01', '2020-01-20'
val_start, val_end = '2020-01-21', '2020-01-25'
test_start, test_end = '2020-01-26', '2020-01-31'

# ========== preprocessed data: select features, create DataFrame, rename columns ========
# making 'XTIME' -> Coordinates and rename to 'time'
ds = ds.swap_dims({'Time': 'XTIME'})
ds = ds.rename({'XTIME': 'time'})

ds = ds[features]
df = ds.to_dataframe()
# save primary df[features] -------------------------------------
df.to_parquet(data_dir_path + 'df_raw.parquet', index=True) # -> eda.py
ds.to_netcdf(data_dir_path + "ds_raw.nc")

# ========== select nodes sn x we, rename columns, add node_id ==============================
selected_ds = ds.isel(south_north=sn, west_east=we)
df = selected_ds.to_dataframe().reset_index()
# south_north, west_east -> node_id (using indexes from sn, we lists: sni-wei)
df['node_id'] = df['south_north'].astype(str) + '-' + df['west_east'].astype(str)

# save clean df in select nodes -------------------------------------
df.to_parquet(data_dir_path + 'df_nodes.parquet', index=False) # -> eda.py
selected_ds.to_netcdf(data_dir_path + "ds_nodes.nc") # -> preprocess.py