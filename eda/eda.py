import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pandas as pd
from config.data.split_config import train_start, train_end
from config.data.features_config import features

# ========= paths =======================
wrf_path = '../data/WRF_output/wrfout_d01_2020-01-01_00:00:00'
reports_path = '../reports/eda/'
eda_report = 'data_info.txt'
wrf_raw_path = '../data/base_processed/full/df_full_wrf.parquet'
wrf_nodes_path = '../data/base_processed/selected/df_selected_eda.parquet'

# =========== parameters =================
target_name = 'T2'

# =========== load data ==============
ds = xr.open_dataset(
    wrf_path,
    engine='netcdf4',
    decode_times=True,
    chunks={'Time': 24},
)

df = pd.read_parquet(wrf_raw_path)
df_nodes = pd.read_parquet(wrf_nodes_path)

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
corr_T2 = corr[target_name].sort_values(ascending=False)

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