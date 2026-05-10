import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pandas as pd
from config.data.split_year_config import train_start, train_end
from config.data.features_config import features

# ========= paths =======================
wrf_path = '../data/WRF_output/year/wrfout_d01_2020-01-01_00:00:00'
reports_path = '../reports/eda/year/'
eda_report = 'data_info.txt'
wrf_raw_path = '../data/preprocessed/year/full/df_full_wrf.parquet'
wrf_nodes_path = '../data/preprocessed/year/selected/df_selected_eda.parquet'
era5_path = '../data/preprocessed/year/selected/df_selected_era5.parquet'

# ========= functions ===============
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


# =========== load data ==============
ds = xr.open_dataset(
    wrf_path,
    engine='netcdf4',
    decode_times=True,
    chunks={'Time': 24},
)

df = pd.read_parquet(wrf_raw_path)
df_nodes = pd.read_parquet(wrf_nodes_path)
df_era5 = pd.read_parquet(era5_path)

# write file about dataset ----------------
print('(1) Запись файла data_info.txt ...\n')
try:
    with open(reports_path+eda_report, 'w', encoding='utf-8') as f:
        f.write('Список всех признаков:\n')
        f.write(', '.join(list(ds.data_vars)))
        f.write('\n\nСписок выбранных признаков: ')
        f.write(', '.join(features))
        f.write(f'\n\nВид индексов: {df.index}')
        f.write(f'\n\nОбщее количество точек сетки: 58 x 46 (south_north x west_east)')
        f.write(f'\n\nРазмерность датасета: {df.shape}')
        f.write(f'\n\nКоличество взятых узлов: {df_nodes["node_id"].nunique()}')
        f.write(f'\n\nРазмерность после отбора узлов: {df_nodes.shape}')
        f.write(f'\n\nТипы данных:\n{df_nodes.dtypes}')
    print('(1) data_info.txt успешно!\n')
except Exception as e:
    print(f'(1) Ошибка при записи data_info.txt! {e}')

# ============ create train datasets for analysis ========
df_train = df_nodes[(df_nodes['time'] >= train_start) & (df_nodes['time'] <= train_end)]
df_train_era5 =  df_era5[(df_era5['time'] >= train_start) & (df_era5['time'] <= train_end)]

# =========== train data analysis ======================
df_center = df_train[df_train['node_id']=='2-2'].copy() # take center node (29, 23)
df_center = df_center.sort_values('time')
df_center = df_center.set_index('time')

# features hist in center node
df_center[features].hist(bins=20, figsize=(15, 10))
plt.suptitle('Распределение признаков (train)')
plt.savefig(reports_path+'features_hist.png', dpi=300, bbox_inches='tight')
plt.close()

corr = df_center[features].corr(numeric_only=True)
corr_T2 = corr['T2'].sort_values(ascending=False)

# autocorrelation for analyse window size
autocorr_4 = df_center['T2'].autocorr(lag=4)
autocorr_8 = df_center['T2'].autocorr(lag=8)
autocorr_12 = df_center['T2'].autocorr(lag=12)
autocorr_24 = df_center['T2'].autocorr(lag=24)

# write file about dataset -----------------------
print('(2) Запись файла data_info.txt ...\n')
try:
    with open(reports_path+eda_report, 'a', encoding='utf-8') as f:
        f.write(f'\n\n\nОбзор train:\n\n')
        df_train.info(buf=f)
        f.write(f'\n\n{df_train.head().to_string()}')
        f.write(f'\n\nСводная статистика train:\n{df_train.describe().to_string()}')
        f.write(f'\n\nПропущенные значения в train:\n{df_train.isna().sum()}')
        f.write(f'\n\nКорреляции целевой переменной T2:\n{corr_T2}')
        f.write(f'\n\nАвтокорреляция\nпри lag = 4 : {autocorr_4}')
        f.write(f'\nпри lag = 8 : {autocorr_8}')
        f.write(f'\nпри lag = 12 : {autocorr_12}')
        f.write(f'\nпри lag = 24 : {autocorr_24}\n')
    print('(2) data_info.txt успешно!\n')
except Exception as e:
    print(f'(2) Ошибка при записи data_info.txt! {e}')

# corr matrix in center node
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица (train)')
plt.savefig(reports_path+'corr_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# spatial-temporal T2 analysis ===========================
df_train["season"] = df_train['time'].dt.month.map(get_season)
df_train_era5["season"] = df_train_era5['time'].dt.month.map(get_season)

seasons = ["Winter", "Spring", "Summer", "Autumn"]

# Spatial mean T2 map
fig, axes = plt.subplots(4, 2, figsize=(14, 20), constrained_layout=True)

for i, season in enumerate(seasons):
    # WRF
    wrf_season = df_train[df_train["season"] == season]
    T2_wrf_mean = wrf_season.groupby(["south_north", "west_east"])["T2"].mean().unstack()

    im0 = axes[i, 0].imshow(T2_wrf_mean.values, cmap="coolwarm", vmin=250, vmax=300)

    axes[i, 0].set_title(f"WRF mean temperature ({season})")
    axes[i, 0].set_xlabel("Longitude index")
    axes[i, 0].set_ylabel("Latitude index")

    # ERA5
    era5_season = df_train_era5[df_train_era5["season"] == season]
    T2_era5_mean = era5_season.groupby(["south_north", "west_east"])["t2m"].mean().unstack()

    im1 = axes[i, 1].imshow(T2_era5_mean.values, cmap="coolwarm", vmin=250, vmax=300)

    axes[i, 1].set_title(f"ERA5 mean temperature ({season})")
    axes[i, 1].set_xlabel("Longitude index")
    axes[i, 1].set_ylabel("Latitude index")


    fig.colorbar(im0, ax=axes[i], label="Temperature (K)", fraction=0.02, pad=0.02)

plt.savefig(reports_path + "seasonal_mean_temperature_map.png")
plt.close()


# temporal T2 dynamics
T2_wrf_time = df_train.groupby('time')['T2'].mean()
T2_era5_time = df_train_era5.groupby('time')['t2m'].mean()


plt.figure(figsize=(25,5))

plt.plot(T2_wrf_time.index, T2_wrf_time.values, label='WRF')
plt.plot(T2_era5_time.index, T2_era5_time.values, label='ERA5')

plt.xlabel("Time step")
plt.ylabel("Temperature (K)")
plt.title("Mean temperature temporal evolution")

plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(reports_path+"temperature_timeseries.png")
plt.close()

# Spatial variability map (std map)
fig, axes = plt.subplots(4, 2, figsize=(14, 20), constrained_layout=True)
vmin_std = 0
vmax_std = 11
for i, season in enumerate(seasons):
    # WRF
    wrf_season = df_train[df_train["season"] == season]
    T2_wrf_std = wrf_season.groupby(["south_north", "west_east"])["T2"].std().unstack()

    im2 = axes[i, 0].imshow(T2_wrf_std.values, cmap="viridis", vmin=vmin_std, vmax=vmax_std)

    axes[i, 0].set_title(f"WRF temperature std ({season})")
    axes[i, 0].set_xlabel("Longitude index")
    axes[i, 0].set_ylabel("Latitude index")

    # ERA5
    era5_season = df_train_era5[df_train_era5["season"] == season]
    T2_era5_std = era5_season.groupby(["south_north", "west_east"])["t2m"].std().unstack()

    im3 = axes[i, 1].imshow(T2_era5_std.values, cmap="viridis", vmin=vmin_std, vmax=vmax_std)

    axes[i, 1].set_title(f"ERA5 temperature std ({season})")
    axes[i, 1].set_xlabel("Longitude index")
    axes[i, 1].set_ylabel("Latitude index")


    fig.colorbar(im2, ax=axes[i], label="Temperature std (K)", fraction=0.02, pad=0.02)

plt.savefig(reports_path + "seasonal_temperature_std_map.png")
plt.close()