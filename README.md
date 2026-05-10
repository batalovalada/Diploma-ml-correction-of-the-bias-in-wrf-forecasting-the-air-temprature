# Коррекция систематической ошибки прогноза температуры воздуха модели WRF на основе методов машинного обучения
## Введение
Задачи метеорологического прогнозирования представляют особую значимость в различных областях человеческой деятельности,
включая сельское хозяйство, энергетику, транспорт и системы предупреждения чрезвычайных ситуаций. Качество принимаемых решений в 
этих сферах напрямую зависит от точности и заблаговременности метеорологических прогнозов.

Несмотря на высокий уровень развития численных моделей, таких как Weather Research and Forecasting Model (WRF), их применение
сопровождается рядом ограничений. К числу основных относятся чувствительность к ошибкам начальных и граничных условий, приближённый характер физических 
параметризаций, а также ограниченное пространственное и временное разрешение модели. Это приводит к возникновению систематических 
ошибок прогноза, в частности при моделировании приземной температуры воздуха, что существенно снижает практическую ценность прогностической 
информации.

Одним из перспективных направлений повышения качества численных прогнозов является применение методов машинного обучения для 
статистической постобработки результатов моделирования. Использование гибридных подходов, сочетающих физически обоснованные численные модели и алгоритмы
машинного обучения, позволяет выполнять коррекцию систематических ошибок прогноза и повышать точность прогнозируемых метеорологических параметров.

## Постановка задачи
Целью данного исследования является разработка и экспериментальная оценка методов машинного обучения в задаче коррекции систематической 
ошибки прогноза температуры воздуха на высоте 2 м (T2), получаемого с использованием модели WRF.

В рамках исследования решаются следующие задачи:
- Выполнить численный ретроспективный прогноз метеорологических параметров моделью WRF на основе данных реанализа ERA5. 
- Подготовить и синхронизировать данные численного моделирования и реанализа ERA5 для последующего применения методов постобработки.
- Реализовать методы машинного обучения для коррекции систематической ошибки прогноза температуры воздуха.
- Выполнить сравнительный анализ эффективности выбранных методов коррекции на основе статистических метрик качества прогноза.
- Выявить модели, показывающие наилучшие результаты в задаче коррекции прогноза приземной температуры воздуха на региональном масштабе.

## Структура проекта
```text
project/
├── config/                    # Файлы конфигураций
│   ├── data/                  # Названия признаков и временные границы split
│   ├── hyperparameters/       # Гиперпараметры ML-моделей
│   └── wrf/                   # Конфигурации WPS, WRF
│
├── data/                      # Данные
│   ├── ConvLSTM/              # ConvLSTM датасет
│   ├── ERA5/                  # Исходные данные ERA5
│   ├── preprocessed/          # Результаты базовой предобработки данных
│   ├── tree_models/           # Random Forest / XGBoost датасет
│   └── WRF_output/            # Результаты моделирования WRF
│   
├── eda/                       # Разведочный анализ данных (EDA)
│   
├── metrics_utils/             # Метрики оценки
│   
├── models/                    # Модели постобработки и реализации их пайпланов
│   ├── ConvLSTM/              # Модель ConvLSTM
│   └── tree_models/           # Модели Random Forest и XGBoost
│       ├── RF/
│       └── XGB/
│   
├── preprocessing/             # Базовая предобработка данных
│   
├── reports/                   # Отчёты и результаты
│   ├── eda/                   # Результаты EDA
│   └── models/                # Результаты коррекции прогноза ML-моделями
│       ├── ConvLSTM/
│       ├── RF/
│       └── XGB/
│   
├── scripts/                   # Скрипты запуска WRF и скачивания данных ERA5 
│   
├── visualization/             # Построение графиков
│   
├── requirements.txt           # Установка зависимостей в проект
│
├── .gitignore
└── README.md
```

## Используемые технологии
В работе используются следующие технологии и инструменты (дата обращения: 04.05.2026):

### Численное моделирование атмосферы
- WRF Preprocessing System (WPS): https://github.com/wrf-model/WPS
- WRF Model (Weather Research and Forecasting Model, ARW core): https://github.com/wrf-model/WRF

### Библиотеки и системные зависимости для численного моделирования
- HDF5 data model and library: https://github.com/HDFGroup/hdf5/archive/hdf5-1_10_5.tar.gz
- JasPer JPEG2000 library: https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/jasper-1.900.1.tar.gz
- libpng library: https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/libpng-1.2.50.tar.gz
- MPICH: High-Performance Message Passing Library: https://www.mpich.org/static/downloads/4.2.2/mpich4.2.2.tar.gz
- NetCDF-C library: https://github.com/Unidata/netcdf-c/archive/v4.7.2.tar.gz
- NetCDF-Fortran library: https://github.com/Unidata/netcdf-fortran/archive/v4.5.2.tar.gz
- Zlib compression library: https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/zlib-1.2.11.tar.gz
- GCC, GFortran, Make, CMake

### Машинное обучение
- Random Forest
- XGBoost
- ConvLSTM

### Python-библиотеки
Основные:
- NumPy
- Pandas
- Xarray
- Scikit-learn
- PyTorch
- Matplotlib
- Optuna

Полный список с версиями в requirements.txt.

## Описание данных
### Источники данных
Реанализ ERA5 (ECMWF Copernicus Climate Data Store) для surface-levels, pressure-levels с ресурсов (дата обращения: 04.05.2026):

- pressure-levels: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview,
- surface-levels: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview.

### Область исследования
Область исследования: Восточно-Балтийский регион (Финский залив и прилегающие территории), ограниченный координатами 62.5°–57.5° с.ш. и 27.0°–35.0° в.д.

Временой интервал: 01.01.2020 (00:00) - 31.12.2020 (21:00).

Временной шаг: 3 часа.

### Типы данных
- ERA5 (GRIB): pressure level (геопотенциал, температура, ветер, влажность), surface level (температура воздуха на высоте 2 м, давление, почва, ветер и др.).
- WRF output (NetCDF: wrfout_d01_*).

### Хранение данных
- data/ERA5/ — исходные данные реанализа
- data/WRF_output/ — результаты моделирования WRF
- data/preprocessed/ — синхронизированные и подготовленные данные

## Схема этапов работы
ERA5 -> WPS -> WRF -> preprocessing -> ML correction -> metrics -> visualization

## Инструкция запуска
1. Установка системных зависимостей: HDF5, JasPer, libpng, MPICH, NetCDF-C, NetCDF-Fortran, Zlib.

А также прочие глобальные зависимости:

```bash
sudo apt install -y \
build-essential gfortran gcc g++ make cmake m4 perl csh git wget \
libnetcdf-dev libnetcdff-dev netcdf-bin libhdf5-dev libhdf5-openmpi-dev \
libpng-dev zlib1g-dev libopenjp2-7-dev openmpi-bin libopenmpi-dev libcurl4-openssl-dev
```

2. Установка WRF и WPS.
3. Установка Python-зависимостей
```bash
pip install -r requirements.txt
```

4. Подготовка данных ERA5. Скачивание выполняется скриптами: scripts/download_era5_year.py, scripts/download_era5_month.py.
5. Подготовка географических данных о характере поверхности с ресурса: https://www2.mmm.ucar.edu/wrf/src/wps_files/geog_high_res_mandatory.tar.gz (используются только для моделирования).
6. Запуск WRF pipeline. Единый запуск: scripts/run_pipeline.sh. Он последовательно выполняет:

- предобработку данных: run_wps.sh -> geogrid.exe / ungrib.exe / metgrid.exe,
- определение начальных и граничных условий: run_real.sh -> real.exe,
- моделирование атмосферных параметров: run_wrf.sh -> wrf.exe.
7. Ретроспективное моделирование WRF

- период: 2020 год
- временной шаг: 3 часа
- пространственное разрешение: 9 км

Моделирование выполняется по месяцам с изменением: start_date, end_date в конфигурациях WPS/WRF.
8. Обработка и ML-пайплайн:

- синхронизация ERA5 и WRF
- формирование признаков
- выбор 25 контрольных точек
- разведочный анализ данных
- обучение моделей Random Forest (RF), XGBoost (XGB), ConvLSTM
9. Обучение и оценка моделей. Скрипты:

- models/ConvLSTM/train.py
- models/tree_models/train.py

Метрики: RMSE, MAE, Bias, Pearson correlation (r).
10. Результирующие метрики оценки, графики обучения и результатов коррекции хранятся в reports/models/.  
