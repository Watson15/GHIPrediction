import math

import numpy as np
import pandas as pd
import pytest
from dataRetrieval import (
    adjustWindDirection,
    chunk,
    compute_angle,
    euclidean_distance,
    find_nearest_station_given_long_lat,
    get_relative_angle_degrees,
    getEastWestNorthSouthOrder,
    getMaxStartMinEndYearComplete,
    latlon_to_xy,
    spatially_diverse_knn,
)


def test_chunk_basic():
    df = pd.DataFrame({"A": range(30)})
    result = chunk(df, interval=5)

    # Should be (30-5) = 25 chunks
    assert result.shape == (25, 5, 1)

    # First chunk should be rows 0–4
    assert result[0].flatten().tolist() == [0, 1, 2, 3, 4]

    # Second chunk should start at row 1
    assert result[1].flatten().tolist() == [1, 2, 3, 4, 5]


def test_chunk_interval_equal_length():
    df = pd.DataFrame({"A": range(25)})
    result = chunk(df, interval=25)

    # len(df) - interval = 0 → expect empty array
    assert result.shape == (0,)


def test_chunk_interval_larger_than_df():
    df = pd.DataFrame({"A": range(10)})
    result = chunk(df, interval=20)

    # No chunks can be formed
    assert result.shape == (0,)


def test_chunk_single_column_multiple_rows():
    df = pd.DataFrame({"A": [10, 20, 30, 40, 50]})
    result = chunk(df, interval=3)

    # len(df)-3 = 2 → expect 2 chunks
    assert len(result) == 2
    assert result[0].flatten().tolist() == [10, 20, 30]
    assert result[1].flatten().tolist() == [20, 30, 40]


@pytest.mark.parametrize("interval", [1, 2, 4])
def test_chunk_various_intervals(interval):
    df = pd.DataFrame({"A": range(10)})
    result = chunk(df, interval)

    expected_num_chunks = max(0, len(df) - interval)
    assert len(result) == expected_num_chunks

    # Check first chunk correctness (if exists)
    if expected_num_chunks > 0:
        assert result[0].flatten().tolist() == list(range(interval))

@pytest.mark.parametrize(
    "start_year,end_year,expected_start,expected_end",
    [
        (2005, 2006, 2005010100, 2006123124),
        (1980, 1985, 1980010100, 1985123124),
        (2024, 2024, 2024010100, 2024123124),
    ],
)
def test_getMaxStartMinEndYearComplete(start_year, end_year, expected_start, expected_end):
    start, end = getMaxStartMinEndYearComplete(start_year, end_year)
    assert start == expected_start
    assert end == expected_end


@pytest.mark.parametrize(
    "wind_dir, theta_rel, expected",
    [
        # Simple subtraction
        (100, 30, 70),

        # Wrap negative to 0–360
        (20, 60, 320),

        # Large wrap beyond 360
        (400, 30, 370 % 360),     # 370 → 10

        # Exact 360 wrap should become 0
        (360, 0, 0),

        # Subtraction resulting in exactly 360
        (10, -350, 0),            # 10 - (-350) = 360 → 0
    ]
)
def test_adjustWindDirection_scalar(wind_dir, theta_rel, expected):
    result = adjustWindDirection(wind_dir, theta_rel)
    assert result == expected, f"Expected {expected}, got {result}"
    

def test_adjustWindDirection_array_input():
    """
    Ensure the function works with NumPy arrays or list-like inputs.
    """
    wind_dirs = np.array([10, 200, 350])
    theta_rel = 30

    result = adjustWindDirection(wind_dirs, theta_rel)

    expected = (wind_dirs - theta_rel + 360) % 360

    assert np.allclose(result, expected), "Array wind direction adjustment failed"

@pytest.mark.parametrize(
    "vec1, vec2, expected_angle",
    [
        ((1, 0), (0, 1), math.pi/2),
        ((0, 1), (-1, 0), math.pi/2),
        ((-1, 0), (0, -1), math.pi/2),
        ((0, -1), (1, 0), math.pi/2),
        ((0,0), (0,0), 0)
    ]
)
def test_ComputeAngle(vec1,vec2, expected_angle):
    angle = compute_angle(vec1, vec2)
    assert math.isclose(angle, expected_angle, abs_tol=1e-5)

def test_spatially_diverse_knn_basic():
    """
    Tests spatially_diverse_knn using a synthetic dataset of stations
    arranged around a central station in multiple directions.
    Ensures:
    - Correct k neighbors returned
    - Neighbors are spatially diverse (different directions)
    - Center station is excluded
    - Ordering East -> West -> North -> South is respected
    """

    # Create synthetic dataset around center station "Center"
    data = {
        "station": ["Center", "East1", "West1", "North1", "South1", "NE1", "NW1", "SE1", "SW1"],
        "Latitude":  [50, 50, 50, 51, 49, 51, 51, 49, 49],
        "Longitude": [-100, -99, -101, -100, -100, -99, -101, -99, -101],
    }
    df = pd.DataFrame(data)

    # k neighbors to choose
    k = 4

    result_df, center_latlon = spatially_diverse_knn(df, "Center", k=k, candidate_pool=20)

    # --- Basic shape tests ---
    assert len(result_df) == k, f"Expected {k} neighbors, got {len(result_df)}"

    # Ensure center is not included
    assert "Center" not in result_df["station"].tolist()

    # Ensure the returned stations exist in the dataset
    for s in result_df["station"]:
        assert s in df["station"].values

    # --- Test angular diversity ---
    # Get center XY using the same conversion
    lat0 = center_latlon[0]
    cx, cy = latlon_to_xy(center_latlon[0], center_latlon[1], lat0)
    center_xy = np.array([cx, cy])

    # Compute vectors for all selected neighbors
    station_vectors = []
    for _, row in result_df.iterrows():
        vx, vy = latlon_to_xy(row["Latitude"], row["Longitude"], lat0)
        station_vectors.append(np.array([vx, vy]) - center_xy)

    # Check that no two selected vectors are too close in direction (angle < 30°)
    # Note: spatially diverse_knn uses a 45° threshold internally
    for i in range(len(station_vectors)):
        for j in range(i + 1, len(station_vectors)):
            angle = compute_angle(station_vectors[i], station_vectors[j])
            assert angle >= np.radians(25), (
                f"Selected neighbors are not spatially diverse enough: angle={np.degrees(angle)}°"
            )

    # --- Test ordering: East -> West -> North -> South ---
    x = result_df["x"]
    y = result_df["y"]
    # East has highest x
    assert result_df.iloc[0]["x"] == max(x)
    # West has lowest x
    assert result_df.iloc[1]["x"] == min(x)
    # North has highest y among remaining
    assert result_df.iloc[2]["y"] == max(y.iloc[2:])
    # South has lowest y among remaining
    assert result_df.iloc[3]["y"] == min(y.iloc[3:])


def test_spatially_diverse_knn_station_not_found():
    df = pd.DataFrame({
        "station": ["A", "B"],
        "Latitude": [1, 2],
        "Longitude": [3, 4]
    })

    result = spatially_diverse_knn(df, "MissingStation", k=3)
    assert isinstance(result, str)
    assert "not found" in result

def test_find_nearest_station_given_long_lat():
    # Create synthetic dataset
    df = pd.DataFrame({
        "station": ["A", "B", "C"],
        "Latitude": [50.0, 50.1, 49.9],
        "Longitude": [-100.0, -100.05, -99.95],
        "StartTime": [1, 1, 1],
        "EndTime": [2, 2, 2]
    })

    # Target location near station A
    target_lat = 50.02
    target_lon = -100.01

    nearest = find_nearest_station_given_long_lat(df.copy(), target_lon, target_lat)

    # --- Basic shape tests ---
    assert len(nearest) == 1, "Function should return exactly one row"

    station_name = nearest.iloc[0]["station"]
    assert station_name == "A", f"Expected station A as nearest, got {station_name}"

    # --- Columns returned ---
    expected_cols = ['station', 'Latitude', 'Longitude', 'distance', 'StartTime', 'EndTime']
    assert list(nearest.columns) == expected_cols, "Returned columns do not match expected list"

    # --- Validate distance value ---
    computed_dist, _ = euclidean_distance(
        target_lat,
        target_lon,
        df.loc[df["station"] == "A", "Latitude"].iloc[0],
        df.loc[df["station"] == "A", "Longitude"].iloc[0],
    )
    returned_dist = nearest.iloc[0]["distance"][0]  # distance tuple: (value, (dx,dy))

    assert np.isclose(computed_dist, returned_dist, atol=1e-6), \
        "Distance value does not match expected Euclidean distance"


def test_find_nearest_station_tie_break():
    """
    If two stations are equidistant, function should still return exactly 1 row.
    """
    df = pd.DataFrame({
        "station": ["A", "B"],
        "Latitude": [50.0, 50.0],
        "Longitude": [-100.0, -100.0],  # EXACT same location
        "StartTime": [1, 1],
        "EndTime": [2, 2]
    })

    nearest = find_nearest_station_given_long_lat(df.copy(), -100.0, 50.0)

    assert len(nearest) == 1, "Even with a tie, should return exactly one row"
    assert nearest.iloc[0]["station"] in ["A", "B"], "Nearest station should be one of the tied stations"

def test_relative_angles_basic():
    """
    Tests relative angles for simple points in each quadrant.
    """

    # Create synthetic DataFrame with distance vectors
    # Format of 'distance': (magnitude, (dx, dy))
    data = {
        "station": ["East", "North", "West", "South", "NE", "SW"],
        "distance": [
            (1, (1, 0)),    # East → 0°
            (1, (0, 1)),    # North → 90°
            (1, (-1, 0)),   # West → 180°
            (1, (0, -1)),   # South → 270°
            (1, (1, 1)),    # NE → 45°
            (1, (-1, -1)),  # SW → 225°
        ]
    }

    df = pd.DataFrame(data)

    result = get_relative_angle_degrees(df)

    expected = [0, 90, 180, 270, 45, 225]

    # Allow small floating-point tolerance
    assert all(np.isclose(r, e) for r, e in zip(result, expected)), f"Expected {expected}, got {result}"


def test_relative_angles_zero_vector():
    """
    Zero-length vector should return 0° by default (arctan2(0,0)=0).
    """
    df = pd.DataFrame({
        "station": ["Center"],
        "distance": [(0, (0, 0))]
    })

    result = get_relative_angle_degrees(df)

    assert result[0] == 0, f"Expected 0°, got {result[0]}"


def test_relative_angles_multiple_stations():
    """
    Test multiple arbitrary vectors.
    """
    df = pd.DataFrame({
        "station": ["A", "B"],
        "distance": [
            (5, (3, 4)),  # arctan2(4,3) → 53.1301°
            (2, (-1, 1))  # arctan2(1,-1) → 135°
        ]
    })

    result = get_relative_angle_degrees(df)

    expected = [np.degrees(np.arctan2(4, 3)), np.degrees(np.arctan2(1, -1))]
    expected = [e if e >= 0 else e + 360 for e in expected]

    assert all(np.isclose(r, e) for r, e in zip(result, expected)), f"Expected {expected}, got {result}"


def test_GetEastWestNorthSouthOrder():
    # Create a small test DataFrame with known positions
    data = {
        "station": ["A", "B", "C", "D"],
        "x":       [10,   -5,   3,    1],   # East-West axis
        "y":       [2,     9,  -4,    7],   # North-South axis
    }
    df = pd.DataFrame(data)

    # Run function
    result = getEastWestNorthSouthOrder(df)

    # Extract station order
    ordered = list(result["station"])

    # Expected:
    # East = max x = station A (x=10)
    # West = min x = station B (x=-5)
    # North = max y of remaining C/D = station D (y=7)
    # South = min y of remaining C   = station C (y=-4)
    expected = ["A", "B", "D", "C"]

    assert ordered == expected, f"Expected {expected}, got {ordered}"