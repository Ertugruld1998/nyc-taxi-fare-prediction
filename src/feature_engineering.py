"""
Feature Engineering Module for NYC Taxi Fare Prediction (FIXED)
Creates predictive features with validation
"""

import pandas as pd
import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points in miles"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return 3959 * c  # Earth radius in miles


def manhattan_distance(lat1, lon1, lat2, lon2):
    """Calculate Manhattan (grid) distance in miles"""
    lat_distance = abs(lat2 - lat1) * 69  # 1 degree latitude ≈ 69 miles
    lon_distance = abs(lon2 - lon1) * 54.6  # 1 degree longitude ≈ 54.6 miles at NYC latitude
    return lat_distance + lon_distance


def bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing (direction) between two points in degrees"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)

    return (initial_bearing + 360) % 360


def create_distance_features(df):
    """
    Create distance-based features with validation
    """
    print("Creating distance features...")

    df['distance_miles'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    df['manhattan_distance'] = manhattan_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    df['bearing'] = bearing(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    # NEW: Distance squared feature (captures non-linear relationship)
    df['distance_squared'] = df['distance_miles'] ** 2

    # NEW: Log distance (helps with scale)
    df['log_distance'] = np.log1p(df['distance_miles'])

    return df


def create_location_features(df):
    """
    Create features based on distances to key locations
    """
    print("Creating location features...")

    # NYC airport coordinates
    jfk = (40.6413, -73.7781)
    lga = (40.7769, -73.8740)
    ewr = (40.6895, -74.1745)
    manhattan_center = (40.7580, -73.9855)  # Times Square

    # Distance to JFK Airport
    df['pickup_distance_to_jfk'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'], jfk[0], jfk[1]
    )
    df['dropoff_distance_to_jfk'] = haversine_distance(
        df['dropoff_latitude'], df['dropoff_longitude'], jfk[0], jfk[1]
    )

    # Distance to LaGuardia Airport
    df['pickup_distance_to_lga'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'], lga[0], lga[1]
    )
    df['dropoff_distance_to_lga'] = haversine_distance(
        df['dropoff_latitude'], df['dropoff_longitude'], lga[0], lga[1]
    )

    # Distance to Newark Airport
    df['pickup_distance_to_ewr'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'], ewr[0], ewr[1]
    )
    df['dropoff_distance_to_ewr'] = haversine_distance(
        df['dropoff_latitude'], df['dropoff_longitude'], ewr[0], ewr[1]
    )

    # Distance to Manhattan center
    df['distance_to_center'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        manhattan_center[0], manhattan_center[1]
    )

    # NEW: Check if pickup or dropoff is near airport (within 1 mile)
    df['is_airport_trip'] = (
        (df['pickup_distance_to_jfk'] < 1) | (df['dropoff_distance_to_jfk'] < 1) |
        (df['pickup_distance_to_lga'] < 1) | (df['dropoff_distance_to_lga'] < 1) |
        (df['pickup_distance_to_ewr'] < 1) | (df['dropoff_distance_to_ewr'] < 1)
    ).astype(int)

    return df


def validate_features(df):
    """
    NEW: Validate all features are reasonable
    """
    print("Validating features...")

    # Check for any NaN or infinite values
    if df.isnull().any().any():
        print("⚠️  WARNING: Found NaN values in features")
        df = df.dropna()

    if np.isinf(df.select_dtypes(include=[np.number]).values).any():
        print("⚠️  WARNING: Found infinite values in features")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure distances are positive
    distance_cols = ['distance_miles', 'manhattan_distance', 'log_distance']
    for col in distance_cols:
        if col in df.columns:
            if (df[col] < 0).any():
                print(f"⚠️  WARNING: Found negative values in {col}")
                df = df[df[col] >= 0]

    return df


def engineer_features(df):
    """
    Complete feature engineering pipeline with validation
    """
    print("\n=== Starting Feature Engineering ===\n")

    df = create_distance_features(df)
    df = create_location_features(df)

    # Remove zero-distance trips (more strict)
    initial_count = len(df)
    df = df[df['distance_miles'] > 0.1]  # Changed from >0 to >0.1
    removed = initial_count - len(df)
    print(f"Removed {removed:,} near-zero distance trips (< 0.1 miles)")

    # NEW: Validate all features
    df = validate_features(df)

    print(f"\nFinal feature count: {df.shape[1]}")
    print(f"Distance range: {df['distance_miles'].min():.2f} - {df['distance_miles'].max():.2f} miles")
    print(f"Median distance: {df['distance_miles'].median():.2f} miles")

    # NEW: Calculate and show fare per mile statistics
    df['temp_fare_per_mile'] = df['fare_amount'] / df['distance_miles']
    print(f"\nFare per mile statistics:")
    print(f"  Mean: ${df['temp_fare_per_mile'].mean():.2f}/mile")
    print(f"  Median: ${df['temp_fare_per_mile'].median():.2f}/mile")
    print(f"  Range: ${df['temp_fare_per_mile'].min():.2f} - ${df['temp_fare_per_mile'].max():.2f}/mile")
    df = df.drop('temp_fare_per_mile', axis=1)

    print("=== Feature Engineering Complete ===\n")

    return df