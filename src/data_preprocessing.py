"""
Data Preprocessing Module for NYC Taxi Fare Prediction (FIXED)
Handles data loading, cleaning, and datetime feature extraction
INCLUDES: Relaxed outlier removal for airport trips
"""

import pandas as pd
import numpy as np


def load_data(filepath, nrows=None):
    """
    Load taxi data from CSV file

    Args:
        filepath: Path to CSV file
        nrows: Number of rows to load (None = all data)

    Returns:
        DataFrame with taxi trip data
    """
    print(f"Loading data from {filepath}...")
    if nrows:
        df = pd.read_csv(filepath, nrows=nrows)
        print(f"Loaded {len(df):,} records (limited to first {nrows:,} rows)")
    else:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df):,} records")
    return df


def remove_outliers(df):
    """
    Remove invalid data and outliers using STRICT domain knowledge
    FIXED: Relaxed fare-per-mile limits for airport trips
    """
    print("Removing outliers (STRICT MODE with airport exceptions)...")
    initial_count = len(df)

    # STEP 1: Remove invalid fares (NYC minimum fare is $2.50)
    df = df[(df['fare_amount'] >= 2.5) & (df['fare_amount'] <= 250)]  # Raised max for airport trips

    # STEP 2: Strict NYC coordinate bounds
    df = df[
        (df['pickup_latitude'].between(40.55, 40.92)) &
        (df['pickup_longitude'].between(-74.05, -73.75)) &
        (df['dropoff_latitude'].between(40.55, 40.92)) &
        (df['dropoff_longitude'].between(-74.05, -73.75))
    ]

    # STEP 3: Valid passenger counts (1-6)
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]

    # STEP 4: Calculate preliminary distance to validate trips
    from src.feature_engineering import haversine_distance
    df['temp_distance'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    # Remove trips that are too short or too long
    df = df[(df['temp_distance'] >= 0.1) & (df['temp_distance'] <= 50)]

    # STEP 5: Identify airport trips BEFORE fare-per-mile filtering
    jfk = (40.6413, -73.7781)
    lga = (40.7769, -73.8740)
    ewr = (40.6895, -74.1745)

    df['temp_near_jfk_pickup'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'], jfk[0], jfk[1]
    ) < 1.5
    df['temp_near_jfk_dropoff'] = haversine_distance(
        df['dropoff_latitude'], df['dropoff_longitude'], jfk[0], jfk[1]
    ) < 1.5
    df['temp_near_lga_pickup'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'], lga[0], lga[1]
    ) < 1.5
    df['temp_near_lga_dropoff'] = haversine_distance(
        df['dropoff_latitude'], df['dropoff_longitude'], lga[0], lga[1]
    ) < 1.5
    df['temp_near_ewr_pickup'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'], ewr[0], ewr[1]
    ) < 1.5
    df['temp_near_ewr_dropoff'] = haversine_distance(
        df['dropoff_latitude'], df['dropoff_longitude'], ewr[0], ewr[1]
    ) < 1.5

    df['temp_is_airport'] = (
        df['temp_near_jfk_pickup'] | df['temp_near_jfk_dropoff'] |
        df['temp_near_lga_pickup'] | df['temp_near_lga_dropoff'] |
        df['temp_near_ewr_pickup'] | df['temp_near_ewr_dropoff']
    )

    # STEP 6: Calculate fare per mile with DIFFERENT rules for airport vs regular
    df['temp_fare_per_mile'] = df['fare_amount'] / df['temp_distance']

    # RELAXED limits for airport trips (to keep JFK flat rate trips)
    # STRICT limits for regular trips
    airport_mask = df['temp_is_airport'] & df['temp_fare_per_mile'].between(1.0, 15.0)
    regular_mask = ~df['temp_is_airport'] & df['temp_fare_per_mile'].between(1.5, 10.0)

    df = df[airport_mask | regular_mask]

    airport_kept = airport_mask.sum()
    regular_kept = regular_mask.sum()
    print(f"  Kept {airport_kept:,} airport trips (relaxed: $1-15/mile)")
    print(f"  Kept {regular_kept:,} regular trips (strict: $1.50-10/mile)")

    # Drop temporary columns
    df = df.drop([
        'temp_distance', 'temp_fare_per_mile', 'temp_is_airport',
        'temp_near_jfk_pickup', 'temp_near_jfk_dropoff',
        'temp_near_lga_pickup', 'temp_near_lga_dropoff',
        'temp_near_ewr_pickup', 'temp_near_ewr_dropoff'
    ], axis=1)

    # STEP 7: LESS AGGRESSIVE IQR method (2.0 instead of 1.5 to keep more airport trips)
    Q1 = df['fare_amount'].quantile(0.25)
    Q3 = df['fare_amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.0 * IQR
    upper_bound = Q3 + 2.0 * IQR
    df = df[(df['fare_amount'] >= lower_bound) & (df['fare_amount'] <= upper_bound)]

    removed_count = initial_count - len(df)
    removed_pct = (removed_count / initial_count) * 100
    print(f"Removed {removed_count:,} outliers ({removed_pct:.2f}%)")

    if removed_pct > 50:
        print("⚠️  WARNING: Removed >50% of data. Check if data quality is very poor.")

    return df


def extract_datetime_features(df):
    """
    Extract temporal features from pickup datetime
    """
    print("Extracting datetime features...")

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year

    # Rush hour indicator (7-9 AM and 5-7 PM on weekdays)
    df['is_rush_hour'] = (
        ((df['hour'].between(7, 9) | df['hour'].between(17, 19)) &
         (df['day_of_week'] < 5))
    ).astype(int)

    # Weekend indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Late night indicator (12 AM - 5 AM) - higher fares
    df['is_late_night'] = (df['hour'].between(0, 5)).astype(int)

    return df


def preprocess_data(df):
    """
    Complete preprocessing pipeline with STRICT validation
    """
    print("\n=== Starting Data Preprocessing (STRICT MODE) ===\n")

    print(f"Missing values before cleaning:\n{df.isnull().sum()}\n")
    df = df.dropna()

    df = remove_outliers(df)
    df = extract_datetime_features(df)

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Final fare range: ${df['fare_amount'].min():.2f} - ${df['fare_amount'].max():.2f}")
    print(f"Median fare: ${df['fare_amount'].median():.2f}")
    print("=== Preprocessing Complete ===\n")

    return df