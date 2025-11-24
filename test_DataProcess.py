#from GHIPrediction.processAllData import getStationTimePeriodFromYears, denormalize_non_cyclical, denormalizeHourOrMonth, euclidean_distance, find_k_nearest_stations, latlon_to_xy, makeDistanceTuple, getEastWestNorthSouthOrder, spatially_diverse_knn, find_nearest_station_given_long_lat, find_nearest_station_given_normed_long_lat, modify_nearest_stations, find_stations_csv, adjustWindDirection, cyclicalEncoding, process_data, get_nearest_stations_data, chunk, get_chunked_tensors, getMaxStartMinEndYearComplete, normalize_column, normalize_dataframes, get_relative_angle_degrees, getAllKAuxillaryStationsReadyByWantedStationName, getAllReadyForStationByLatAndLongAndK, getAllReadyForStationByLatAndLongAndKSplitTestAndTrain
import math

import numpy as np
import pandas as pd
from dataRetrieval import normalize_column
from processAllData import (
    cyclicalEncoding,
    denormalize_non_cyclical,
    denormalizeHourOrMonth,
    getStationTimePeriodFromYears,
)


def test_GetStationTimePeriodFromYears():
    csv_file = 'GHIPrediction/Datasets/CWEEDS_2020_BC_raw/CAN_BC_ABBOTSFORD-A_1100031_CWEEDS2011_1998-2017.csv'
    s_year = 1998
    e_year = 2017
    test_s, test_e = getStationTimePeriodFromYears(csv_file)
    assert test_s == s_year
    assert test_e == e_year
    
def __testDenormalize_non_cyclical__(col, mean, std, expected_col):
    denormalized_col = denormalize_non_cyclical(col, mean, std)
    
    # Convert to numpy arrays to compare elementwise
    denormalized_col = np.array(denormalized_col)
    expected_col = np.array(expected_col)
    
    assert np.allclose(denormalized_col, expected_col, atol=1e-5), \
        f"Expected {expected_col}, got {denormalized_col}"

def test_Normalize_column():
    GHI_expected = [0, 250, 500, 750, 1000]
    df = pd.DataFrame({'GHI': GHI_expected})
    df_norm, _, mean, std = normalize_column('GHI', df.copy(), [df.copy()])
    mean_expected = 500
    std_expected = 395.28471 # It is a sample of an infinite population which we wish to make a statement about the entire population. Hence using N-1 in denominator. 
    assert mean == mean_expected
    assert math.isclose(std_expected, std, abs_tol=1e-5)
    __testDenormalize_non_cyclical__(df_norm['GHI'], mean, std, GHI_expected)
    # GHI_norm_expected = [0, 250, 500, 750, 1000]
    # df_norm_expected = pd.DataFrame({'GHI': GHI_norm_expected})
    # assert all(df_norm['GHI'] == df_norm_expected['GHI']) # may work may not

def __testDenormalizeHourOrMonth__(anglesin, anglecos, cycleLength, expected_value):
    denormalized_value = denormalizeHourOrMonth(anglesin, anglecos, cycleLength)
    assert expected_value == denormalized_value, f"Expected {expected_value}, got {denormalized_value}"

def test_CyclicalEncoding():
    hour_expected = [0, 6, 12, 18, 23]
    month_expected = [1, 4, 7, 9, 12]
    df = pd.DataFrame({'hour': hour_expected, 'month': month_expected})

    df["hour_sin"], df["hour_cos"] = cyclicalEncoding(df['hour'], 24)
    
    for row in df.itertuples(index=False):
        __testDenormalizeHourOrMonth__(row.hour_sin, row.hour_cos, 24, row.hour)

    df["month_sin"], df["month_cos"] = cyclicalEncoding(df['month'], 12)

    for row in df.itertuples(index=False):
        __testDenormalizeHourOrMonth__(row.month_sin, row.month_cos, 12, row.month)
    # hour_norm_expected = [0, 6, 12, 18, 23]
    # month_norm_expected = [1,4,7,12]
    # df_norm = pd.DataFrame({'hour': hour_norm_expected, 'month': month_norm_expected})
    





