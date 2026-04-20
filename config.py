CSV_PATH = "./data/33101.csv"
UNIQUE_ID = "ZJ"

INPUT_SIZE = 7
HORIZON = 7
SPLIT_RATIO = (7, 1, 2)
SLIDING_STEP_SIZE = 1
FREQ = "D"

REMOVE_LAST_MONTH = False
SAVE_PLOTS = True
RANDOM_SEED = 2026
EARLY_STOP_PATIENCE_EPOCHS = 10
ML_EARLY_STOPPING_ROUNDS = 20

NEURAL_LOSS_NAME = "MSE"
NEURAL_LOSS_PARAMS = {}
CHECKPOINT_MODE = "last"

PLOT_FORECAST = True
PLOT_LOSS = True
PLOT_LOSS_NAME = "MAE"
SAVE_DIR = "./artifacts"

FUTR_EXOG_LIST = [
    "is_workday",
    "is_holiday",
    "holiday_name_Spring_Festival",
    "sunrise_iso8601",
    "sunset_iso8601",
    "daylight_duration_s",
    "sunrise_iso8601_day_sin",
    "sunrise_iso8601_day_cos",
    "sunset_iso8601_day_sin",
    "sunset_iso8601_day_cos",
    "daylight_duration_s_sin",
    "daylight_duration_s_cos",
]

HIST_EXOG_LIST = [
    "weather_code_wmo_code",
    "temperature_2m_max_degC",
    "temperature_2m_mean_degC",
    "temperature_2m_min_degC",
    "apparent_temperature_max_degC",
    "apparent_temperature_mean_degC",
    "apparent_temperature_min_degC",
    "sunshine_duration_s",
    "uv_index_clear_sky_max_",
    "uv_index_max_",
    "rain_sum_mm",
    "showers_sum_mm",
    "snowfall_sum_cm",
    "precipitation_sum_mm",
    "precipitation_hours_h",
    "et0_fao_evapotranspiration_mm",
    "shortwave_radiation_sum_MJ_m_square",
    "wind_direction_10m_dominant_deg",
    "wind_gusts_10m_max_km_h",
    "wind_speed_10m_max_km_h",
    "sunshine_duration_s_sin",
    "sunshine_duration_s_cos",
    "precipitation_hours_h_sin",
    "precipitation_hours_h_cos",
]
