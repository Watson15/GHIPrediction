COLUMN_NAMES = {
    'GHI': 'GHI_kJ/m2',
    'DNI': 'DNI_kJ/m2',
    'DHI': 'DHI_kJ/m2',
    'YEAR': 'Year',
    'MONTH': 'Month',
    'DAY': 'Day',
    'HOUR': 'Hour',
    'YEAR_MONTH_DAY_HOUR': 'Year Month Day Hour (YYYYMMDDHH)',
    'HOUR_SIN': 'Hour_sin',
    'HOUR_COS': 'Hour_cos',
    'MONTH_SIN': 'Month_sin',
    'MONTH_COS': 'Month_cos',
    'OPAQUE_SKY_COVER': 'opaque_sky_cover',
    'WIND_SPEED': 'wind_speed',
    'WIND_DIRECTION': 'wind_direction',
    'WIND_DIRECTION_SIN': 'wind_dir_sin',
    'WIND_DIRECTION_COS': 'wind_dir_cos',
    'DRY_BULB_TEMPERATURE': 'Dry_Bulb_Temperature_C',
}

USECOLS_CLOUD = [
    COLUMN_NAMES["YEAR_MONTH_DAY_HOUR"],
    COLUMN_NAMES["OPAQUE_SKY_COVER"],
    COLUMN_NAMES["GHI"],
    COLUMN_NAMES["DNI"],
    COLUMN_NAMES["DHI"],
    COLUMN_NAMES["WIND_DIRECTION"],
    COLUMN_NAMES["WIND_DIRECTION_SIN"],
    COLUMN_NAMES["WIND_DIRECTION_COS"],
    COLUMN_NAMES["WIND_SPEED"],
    COLUMN_NAMES["MONTH_SIN"],
    COLUMN_NAMES["MONTH_COS"],
    COLUMN_NAMES["HOUR_COS"],
    COLUMN_NAMES["HOUR_SIN"],
    COLUMN_NAMES["MONTH"],
    COLUMN_NAMES["HOUR"],
]

DTYPE_CLOUD = {
    COLUMN_NAMES["YEAR_MONTH_DAY_HOUR"]: str,
    COLUMN_NAMES["OPAQUE_SKY_COVER"]: str,
    COLUMN_NAMES["GHI"]: str,
    COLUMN_NAMES["DNI"]: str,
    COLUMN_NAMES["DHI"]: str,
    COLUMN_NAMES["WIND_DIRECTION"]: str,
    COLUMN_NAMES["WIND_DIRECTION_SIN"]: str,
    COLUMN_NAMES["WIND_DIRECTION_COS"]: str,
    COLUMN_NAMES["WIND_SPEED"]: str,
    COLUMN_NAMES["MONTH_SIN"]: str,
    COLUMN_NAMES["MONTH_COS"]: str,
    COLUMN_NAMES["HOUR_COS"]: str,
    COLUMN_NAMES["HOUR_SIN"]: str,
    COLUMN_NAMES["MONTH"]: str,
    COLUMN_NAMES["HOUR"]: str,
}

USECOLS_NON_CLOUD = [col for col in USECOLS_CLOUD if col != COLUMN_NAMES["OPAQUE_SKY_COVER"]] #same as cloud but without opaque sky cover

DTYPE_NON_CLOUD = {k: v for k, v in DTYPE_CLOUD.items() if k != COLUMN_NAMES["OPAQUE_SKY_COVER"]} #same as cloud but without opaque sky cover