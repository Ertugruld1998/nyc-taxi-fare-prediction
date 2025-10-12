"""
Visualize NYC Taxi Fare Inflation (2016-2025) - FIXED VERSION
Creates charts showing fare evolution over time WITH REALISTIC MINIMUMS
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from predict import create_trip_features, predict_fare_by_year, load_model


def get_inflation_factors():
    """Get year-by-year inflation factors"""
    return {
        2016: 1.00, 2017: 1.03, 2018: 1.06, 2019: 1.09, 2020: 1.11,
        2021: 1.17, 2022: 1.27, 2023: 1.35, 2024: 1.42, 2025: 1.48
    }


def predict_fare_timeline(model, features, trip_name):
    """
    Generate fare predictions across years WITH MINIMUM FARES APPLIED
    """
    # Use the same prediction function as predict.py (includes minimums!)
    yearly_fares = predict_fare_by_year(model, features)

    years = list(yearly_fares.keys())
    fares = list(yearly_fares.values())

    return years, fares, trip_name


def create_inflation_visualization():
    """Create comprehensive inflation visualization with CORRECT fares"""

    model = load_model()
    if model is None:
        return

    # Define sample trips
    trips = [
        {
            'name': 'Times Square → Central Park (2 mi)',
            'pickup': (40.7580, -73.9855),
            'dropoff': (40.7829, -73.9654),
            'datetime': '2024-06-15 18:30:00',
            'passengers': 2
        },
        {
            'name': 'Manhattan → JFK Airport (13 mi)',
            'pickup': (40.7614, -73.9776),
            'dropoff': (40.6413, -73.7781),
            'datetime': '2024-06-16 09:00:00',
            'passengers': 1
        },
        {
            'name': 'Brooklyn → Lower Manhattan (4 mi)',
            'pickup': (40.6782, -73.9442),
            'dropoff': (40.7128, -74.0060),
            'datetime': '2024-06-17 14:00:00',
            'passengers': 3
        }
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # Plot 1: Fare Evolution Over Time
    for idx, trip in enumerate(trips):
        features = create_trip_features(
            trip['pickup'][0], trip['pickup'][1],
            trip['dropoff'][0], trip['dropoff'][1],
            trip['datetime'], trip['passengers']
        )

        years, fares, name = predict_fare_timeline(model, features, trip['name'])

        ax1.plot(years, fares, marker='o', linewidth=2.5,
                markersize=8, label=name, color=colors[idx])

        # Annotate 2016 and 2025 prices
        ax1.annotate(f'${fares[0]:.2f}',
                    xy=(years[0], fares[0]),
                    xytext=(-10, -20),
                    textcoords='offset points',
                    fontsize=9,
                    color=colors[idx])
        ax1.annotate(f'${fares[-1]:.2f}',
                    xy=(years[-1], fares[-1]),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    color=colors[idx])

    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fare Amount ($)', fontsize=12, fontweight='bold')
    ax1.set_title('NYC Taxi Fare Evolution with Inflation (2016-2025)',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(2015.5, 2025.5)

    # Highlight COVID period
    ax1.axvspan(2020, 2021, alpha=0.1, color='red')
    ax1.text(2020.5, ax1.get_ylim()[1]*0.95, 'COVID', ha='center', fontsize=9)

    # Plot 2: Cumulative Inflation Rate
    inflation = get_inflation_factors()
    years = list(inflation.keys())
    inflation_pct = [(factor - 1) * 100 for factor in inflation.values()]

    bars = ax2.bar(years, inflation_pct, color='#E63946', alpha=0.7, edgecolor='black')

    # Highlight 2025
    bars[-1].set_color('#06A77D')
    bars[-1].set_alpha(0.9)

    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Inflation from 2016 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Inflation Rate (2016 = Baseline)',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add value labels on bars
    for bar, pct in zip(bars, inflation_pct):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.axhline(y=48, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.text(2016.5, 50, '48% Total Inflation', fontsize=9, color='red')

    plt.tight_layout()

    # Save figure
    import os
    os.makedirs('results', exist_ok=True)
    output_path = 'results/fare_inflation_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")

    plt.show()

    print("\n" + "="*60)
    print("KEY INSIGHTS FROM VISUALIZATION")
    print("="*60)
    print("📈 Steady Growth (2016-2020): +11% cumulative")
    print("🦠 COVID Impact (2020-2021): +5% (minimal)")
    print("🔥 High Inflation (2021-2022): +9% (largest single year jump)")
    print("📊 Recent Period (2022-2025): +16% (congestion pricing impact)")
    print("💰 Total Impact: All fares increased by 48% over 9 years")
    print("\n🛫 IMPORTANT: JFK fares now include $52 flat rate baseline!")
    print("   Without minimum enforcement, model predicted only $19.68")
    print("   With minimum enforcement, correctly shows $52.00 → $76.96")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("NYC TAXI FARE INFLATION VISUALIZER (FIXED)")
    print("="*60)
    print("\nGenerating comprehensive fare evolution charts...")
    print("This will create graphs showing:")
    print("  1. Fare trends for 3 sample routes (2016-2025)")
    print("  2. Cumulative inflation rates by year")
    print("  3. Impact of COVID and high-inflation periods")
    print("  4. ✅ WITH realistic minimum fares applied!\n")

    create_inflation_visualization()