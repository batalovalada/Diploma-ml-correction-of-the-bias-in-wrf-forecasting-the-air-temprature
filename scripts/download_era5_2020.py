import cdsapi
import os

c = cdsapi.Client()

YEAR = "2020"

MONTH = "01"


DAYS = [ "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
        ]

AREA = [62.5, 27.0, 57.5, 35.0]  # N, W, S, E

TIMES = ["00:00", "06:00", "12:00", "18:00"]

PRESSURE_LEVELS = [
    "1000", "925", "850", "700", "600",
    "500", "400", "300", "250", "200", "100"
]

os.makedirs("data/ERA5/pressure", exist_ok=True)
os.makedirs("data/ERA5/surface", exist_ok=True)

# =========================
# PRESSURE LEVELS
# =========================

for day in DAYS:
    filename = f"data/ERA5/pressure/ERA5_pressure_{YEAR}_{MONTH}_{day}.grib"

    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping.")
        continue

    print(f"Downloading pressure levels for {YEAR}-{MONTH}-{day}...")

    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "grib",
            "variable": [
                "geopotential",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "specific_humidity"
            ],
            "pressure_level": PRESSURE_LEVELS,
            "year": YEAR,
            "month": MONTH,
            "day": day,
            "time": TIMES,
            "area": AREA
        },
        filename
    )

# =========================
# SURFACE LEVELS
# =========================

for day in DAYS:
    filename = f"data/ERA5/surface/ERA5_surface_{YEAR}_{MONTH}_{day}.grib"

    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping.")
        continue

    print(f"Downloading surface levels for {YEAR}-{MONTH}-{day}...")

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "grib",
            "variable": [
                "2m_temperature",
                "2m_dewpoint_temperature",
                "surface_pressure",
                "geopotential",
                "mean_sea_level_pressure",
                "skin_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "soil_temperature_level_1",
                "soil_temperature_level_2",
                "soil_temperature_level_3",
                "soil_temperature_level_4",
                "volumetric_soil_water_layer_1",
                "volumetric_soil_water_layer_2",
                "volumetric_soil_water_layer_3",
                "volumetric_soil_water_layer_4",
                "snow_depth",
                "orography",
                "land_sea_mask",
            ],
            "year": YEAR,
            "month": MONTH,
            "day": day,
            "time": TIMES,
            "area": AREA
        },
        filename
    )

print("Download completed.")

