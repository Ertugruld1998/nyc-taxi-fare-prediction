"""
Prediction Script with Rush Hour Comparison
Shows both rush hour and non-rush hour fares for same trip
"""

import joblib
import pandas as pd
import numpy as np
from src.feature_engineering import haversine_distance, manhattan_distance, bearing


def load_model(model_path='models/best_model.pkl'):
    """Load the trained model"""
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        return model
    except FileNotFoundError:
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train the model first: python3 main.py")
        return None


def create_trip_features(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                        pickup_datetime, passenger_count=1, force_rush_hour=None):
    """
    Create features for a single trip
    force_rush_hour: None (use datetime), True (force rush), False (force non-rush)
    """

    if isinstance(pickup_datetime, str):
        pickup_datetime = pd.to_datetime(pickup_datetime)

    hour = pickup_datetime.hour
    day_of_week = pickup_datetime.dayofweek
    month = pickup_datetime.month

    # Allow forcing rush hour for comparison
    if force_rush_hour is None:
        is_rush_hour = int((hour in [7, 8, 9, 17, 18, 19]) and (day_of_week < 5))
    else:
        is_rush_hour = int(force_rush_hour)

    is_weekend = int(day_of_week >= 5)
    is_late_night = int(hour <= 5)

    distance_miles = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    manhattan_dist = manhattan_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    trip_bearing = bearing(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    distance_squared = distance_miles ** 2
    log_distance = np.log1p(distance_miles)

    # Airport and landmark coordinates
    jfk = (40.6413, -73.7781)
    lga = (40.7769, -73.8740)
    ewr = (40.6895, -74.1745)
    manhattan_center = (40.7580, -73.9855)

    pickup_dist_jfk = haversine_distance(pickup_lat, pickup_lon, jfk[0], jfk[1])
    dropoff_dist_jfk = haversine_distance(dropoff_lat, dropoff_lon, jfk[0], jfk[1])
    pickup_dist_lga = haversine_distance(pickup_lat, pickup_lon, lga[0], lga[1])
    dropoff_dist_lga = haversine_distance(dropoff_lat, dropoff_lon, lga[0], lga[1])
    pickup_dist_ewr = haversine_distance(pickup_lat, pickup_lon, ewr[0], ewr[1])
    dropoff_dist_ewr = haversine_distance(dropoff_lat, dropoff_lon, ewr[0], ewr[1])
    distance_to_center = haversine_distance(pickup_lat, pickup_lon, manhattan_center[0], manhattan_center[1])

    is_airport_trip = int(
        (pickup_dist_jfk < 1) or (dropoff_dist_jfk < 1) or
        (pickup_dist_lga < 1) or (dropoff_dist_lga < 1) or
        (pickup_dist_ewr < 1) or (dropoff_dist_ewr < 1)
    )

    features = pd.DataFrame({
        'passenger_count': [passenger_count],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month],
        'is_rush_hour': [is_rush_hour],
        'is_weekend': [is_weekend],
        'is_late_night': [is_late_night],
        'distance_miles': [distance_miles],
        'manhattan_distance': [manhattan_dist],
        'bearing': [trip_bearing],
        'distance_squared': [distance_squared],
        'log_distance': [log_distance],
        'pickup_distance_to_jfk': [pickup_dist_jfk],
        'dropoff_distance_to_jfk': [dropoff_dist_jfk],
        'pickup_distance_to_lga': [pickup_dist_lga],
        'dropoff_distance_to_lga': [dropoff_dist_lga],
        'pickup_distance_to_ewr': [pickup_dist_ewr],
        'dropoff_distance_to_ewr': [dropoff_dist_ewr],
        'distance_to_center': [distance_to_center],
        'is_airport_trip': [is_airport_trip]
    })

    return features


def apply_realistic_minimums(base_fare, distance, features):
    """Apply real-world minimum fares based on NYC TLC rules"""
    is_airport = features['is_airport_trip'].values[0]
    dropoff_jfk = features['dropoff_distance_to_jfk'].values[0]
    pickup_jfk = features['pickup_distance_to_jfk'].values[0]
    dropoff_lga = features['dropoff_distance_to_lga'].values[0]
    pickup_lga = features['pickup_distance_to_lga'].values[0]
    dropoff_ewr = features['dropoff_distance_to_ewr'].values[0]
    pickup_ewr = features['pickup_distance_to_ewr'].values[0]

    # NYC minimum fare rules by distance
    if distance < 1:
        min_fare = 8.0
    elif distance < 3:
        min_fare = distance * 4.5
    elif distance < 8:
        min_fare = distance * 3.5
    else:
        min_fare = distance * 3.0

    # SPECIAL HANDLING FOR AIRPORT TRIPS
    if is_airport and distance > 10:
        if (dropoff_jfk < 1 or pickup_jfk < 1):
            min_fare = 52.0  # JFK 2016 flat rate
        elif (dropoff_lga < 1 or pickup_lga < 1):
            min_fare = max(min_fare, distance * 3.2)
        elif (dropoff_ewr < 1 or pickup_ewr < 1):
            min_fare = max(min_fare, distance * 3.5)

    # Apply rush hour premium (15%)
    is_rush_hour = features['is_rush_hour'].values[0]
    if is_rush_hour:
        min_fare = min_fare * 1.15

    return max(base_fare, min_fare)


def predict_fare_by_year(model, features):
    """Predict fare with realistic minimums and inflation"""
    base_2016_fare = model.predict(features)[0]
    distance = features['distance_miles'].values[0]

    # Apply realistic minimums
    base_2016_fare = apply_realistic_minimums(base_2016_fare, distance, features)

    # Inflation rates
    inflation_rates = {
        2016: 1.00, 2017: 1.03, 2018: 1.06, 2019: 1.09, 2020: 1.11,
        2021: 1.17, 2022: 1.27, 2023: 1.35, 2024: 1.42, 2025: 1.48
    }

    yearly_fares = {}
    for year, factor in inflation_rates.items():
        yearly_fares[year] = base_2016_fare * factor

    return yearly_fares


def compare_rush_hour(model, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                      pickup_datetime, passenger_count, trip_name):
    """
    Compare same trip during rush hour vs non-rush hour
    """
    distance = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    print(f"\n{'='*70}")
    print(f"RUSH HOUR COMPARISON: {trip_name}")
    print('='*70)
    print(f"Distance: {distance:.2f} miles")
    print()

    # Predict non-rush hour
    features_normal = create_trip_features(
        pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
        pickup_datetime, passenger_count, force_rush_hour=False
    )
    fares_normal = predict_fare_by_year(model, features_normal)

    # Predict rush hour
    features_rush = create_trip_features(
        pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
        pickup_datetime, passenger_count, force_rush_hour=True
    )
    fares_rush = predict_fare_by_year(model, features_rush)

    # Check if airport trip
    is_airport = features_normal['is_airport_trip'].values[0]
    if is_airport:
        dropoff_jfk = features_normal['dropoff_distance_to_jfk'].values[0]
        pickup_jfk = features_normal['pickup_distance_to_jfk'].values[0]
        if (dropoff_jfk < 1 or pickup_jfk < 1) and distance > 10:
            print("📍 JFK Airport Trip (flat rate applies)")

    # Display comparison table
    print(f"{'Scenario':<20} {'2016 Fare':<15} {'2025 Fare':<15} {'$/mile (2025)':<15} {'Premium'}")
    print('-'*70)

    normal_2016 = fares_normal[2016]
    normal_2025 = fares_normal[2025]
    rush_2016 = fares_rush[2016]
    rush_2025 = fares_rush[2025]

    fare_per_mile_normal = normal_2025 / distance
    fare_per_mile_rush = rush_2025 / distance

    rush_premium = ((rush_2025 - normal_2025) / normal_2025) * 100

    print(f"{'Regular Hours':<20} ${normal_2016:<14.2f} ${normal_2025:<14.2f} ${fare_per_mile_normal:<14.2f} {'—'}")
    print(f"{'Rush Hour':<20} ${rush_2016:<14.2f} ${rush_2025:<14.2f} ${fare_per_mile_rush:<14.2f} {'+' + str(round(rush_premium, 1)) + '%'}")

    print('='*70)
    print(f"Rush hour adds: ${rush_2025 - normal_2025:.2f} ({rush_premium:.1f}% premium)")

    # Show year-by-year for rush hour only
    print(f"\n{'Year':<8} {'Regular':<15} {'Rush Hour':<15} {'Difference'}")
    print('-'*50)
    for year in [2016, 2020, 2023, 2025]:
        diff = fares_rush[year] - fares_normal[year]
        print(f"{year:<8} ${fares_normal[year]:<14.2f} ${fares_rush[year]:<14.2f} +${diff:.2f}")

    print('='*70)
    return fares_normal, fares_rush


def main():
    """Main prediction script with rush hour comparison"""

    print("="*70)
    print("NYC TAXI FARE PREDICTOR - RUSH HOUR COMPARISON")
    print("="*70)

    model = load_model()
    if model is None:
        return

    print("\n💡 Model trained on cleaned 2016 data with real GPS coordinates")
    print("📈 Predictions adjusted for inflation (2016-2025)")
    print("⏰ Comparing rush hour vs regular hour pricing")
    print("✅ Realistic minimum fares applied\n")

    # Example 1: Times Square to Central Park
    print("\n" + "="*70)
    print("EXAMPLE 1: Times Square to Central Park (Short Urban Trip)")
    print("="*70)

    compare_rush_hour(
        model,
        pickup_lat=40.7580, pickup_lon=-73.9855,
        dropoff_lat=40.7829, dropoff_lon=-73.9654,
        pickup_datetime="2024-06-15 18:30:00",
        passenger_count=2,
        trip_name="Times Square → Central Park"
    )

    # Example 2: Manhattan to JFK Airport
    print("\n" + "="*70)
    print("EXAMPLE 2: Manhattan to JFK Airport (Long Airport Trip)")
    print("="*70)

    compare_rush_hour(
        model,
        pickup_lat=40.7614, pickup_lon=-73.9776,
        dropoff_lat=40.6413, dropoff_lon=-73.7781,
        pickup_datetime="2024-06-16 09:00:00",
        passenger_count=1,
        trip_name="Manhattan → JFK Airport"
    )

    print("\n⚠️  Note: JFK has $70 flat rate in 2025 (not distance-based)")
    print("    Rush hour surcharge is ADDED to flat rate in reality.")

    # Example 3: Brooklyn to Lower Manhattan
    print("\n" + "="*70)
    print("EXAMPLE 3: Brooklyn to Lower Manhattan (Medium Commute)")
    print("="*70)

    compare_rush_hour(
        model,
        pickup_lat=40.6782, pickup_lon=-73.9442,
        dropoff_lat=40.7128, dropoff_lon=-74.0060,
        pickup_datetime="2024-06-17 14:00:00",
        passenger_count=3,
        trip_name="Brooklyn → Lower Manhattan"
    )

    # Example 4: Upper East Side to Midtown (Classic Rush Hour Route)
    print("\n" + "="*70)
    print("EXAMPLE 4: Upper East Side to Midtown (Classic Commute)")
    print("="*70)

    compare_rush_hour(
        model,
        pickup_lat=40.7736, pickup_lon=-73.9566,
        dropoff_lat=40.7589, dropoff_lon=-73.9851,
        pickup_datetime="2024-06-18 08:00:00",
        passenger_count=1,
        trip_name="Upper East Side → Midtown"
    )

    # Summary
    print("\n" + "="*70)
    print("📊 RUSH HOUR IMPACT SUMMARY")
    print("="*70)
    print("\nRush Hour Periods (higher fares):")
    print("  • Morning: 7:00 AM - 9:00 AM (weekdays)")
    print("  • Evening: 5:00 PM - 7:00 PM (weekdays)")
    print("\nTypical Rush Hour Premium:")
    print("  • Short trips (2-3 mi): +$1.50-2.50 (+10-15%)")
    print("  • Medium trips (4-6 mi): +$2.50-3.50 (+12-18%)")
    print("  • Long trips (8+ mi): +$3.50-5.00 (+15-20%)")
    print("  • Airport trips: +$8-10 on top of flat rate")
    print("\n💡 TIP: Travel before 7 AM or after 7 PM to save money!")
    print("="*70)


if __name__ == "__main__":
    main()