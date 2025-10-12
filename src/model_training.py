"""
Model Training Module for NYC Taxi Fare Prediction (FIXED)
Trains and evaluates multiple regression models with proper validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


def prepare_features(df):
    """
    Select features for modeling (UPDATED with new features)
    """
    feature_columns = [
        'passenger_count', 'hour', 'day_of_week', 'month',
        'is_rush_hour', 'is_weekend', 'is_late_night',  # Added late night
        'distance_miles', 'manhattan_distance', 'bearing',
        'distance_squared', 'log_distance',  # NEW distance features
        'pickup_distance_to_jfk', 'dropoff_distance_to_jfk',
        'pickup_distance_to_lga', 'dropoff_distance_to_lga',
        'pickup_distance_to_ewr', 'dropoff_distance_to_ewr',
        'distance_to_center',
        'is_airport_trip'  # NEW airport indicator
    ]

    X = df[feature_columns]
    y = df['fare_amount']

    return X, y


def train_models(X_train, y_train):
    """
    Train multiple regression models with IMPROVED hyperparameters
    """
    print("Training models...")

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=10.0),  # Increased regularization
        'Random Forest': RandomForestRegressor(
            n_estimators=150,        # More trees (was 100)
            max_depth=20,            # Deeper trees (was 15)
            min_samples_split=15,    # More samples required (was 10)
            min_samples_leaf=6,      # More samples in leaves (was 4)
            max_features='sqrt',     # Use sqrt of features (prevents overfitting)
            random_state=42,
            n_jobs=-1,               # Use all CPU cores (was 2)
            verbose=0
        ),
        'Gradient Boosting': GradientBoostingRegressor(  # NEW model
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=15,
            min_samples_leaf=6,
            random_state=42,
            verbose=0
        )
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} training complete")

    return trained_models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models and return metrics with ADDITIONAL validation
    """
    print("\n=== Model Evaluation ===\n")

    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # NEW: Check prediction reasonableness
        # Calculate distance for test set to check fare/mile
        distances = X_test['distance_miles']
        fare_per_mile = y_pred / distances

        # Count unreasonable predictions (< $1/mile or > $15/mile)
        unreasonable_count = ((fare_per_mile < 1) | (fare_per_mile > 15)).sum()
        unreasonable_pct = (unreasonable_count / len(y_pred)) * 100

        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2,
            'Unreasonable_Predictions_%': unreasonable_pct  # NEW metric
        })

        print(f"{name}:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE: ${mae:.2f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Unreasonable predictions: {unreasonable_pct:.2f}%")

        if unreasonable_pct > 5:
            print(f"  ⚠️  WARNING: >5% of predictions are unreasonable!")
        print()

    return pd.DataFrame(results)


def plot_predictions(models, X_test, y_test, output_dir='results'):
    """
    Create prediction scatter plots with fare/mile validation
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)

        # Sample for plotting
        sample_size = min(5000, len(y_test))
        sample_indices = np.random.choice(len(y_test), sample_size, replace=False)

        axes[idx].scatter(
            y_test.iloc[sample_indices],
            y_pred[sample_indices],
            alpha=0.3,
            s=10,
            c='blue'
        )
        axes[idx].plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect prediction')
        axes[idx].set_xlabel('Actual Fare ($)', fontsize=12)
        axes[idx].set_ylabel('Predicted Fare ($)', fontsize=12)
        axes[idx].set_title(f'{name}\nRMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}',
                           fontsize=14)
        axes[idx].set_xlim([0, 80])
        axes[idx].set_ylim([0, 80])
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_predictions.png', dpi=300, bbox_inches='tight')
    print(f"Prediction plots saved to {output_dir}/model_predictions.png")
    plt.close()


def plot_feature_importance(model, feature_names, output_dir='results'):
    """
    Plot feature importance for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features

        plt.figure(figsize=(12, 7))
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)),
                   [feature_names[i] for i in indices],
                   rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {output_dir}/feature_importance.png")
        plt.close()


def plot_fare_vs_distance(X_test, y_test, models, output_dir='results'):
    """
    NEW: Plot actual vs predicted fare as function of distance
    """
    plt.figure(figsize=(14, 8))

    # Sample data
    sample_size = min(3000, len(y_test))
    sample_indices = np.random.choice(len(y_test), sample_size, replace=False)

    distances = X_test['distance_miles'].iloc[sample_indices]
    actual_fares = y_test.iloc[sample_indices]

    # Plot actual fares
    plt.scatter(distances, actual_fares, alpha=0.3, s=20,
                label='Actual', c='blue')

    # Plot predictions from best model (Random Forest or Gradient Boosting)
    best_model_name = 'Random Forest' if 'Random Forest' in models else list(models.keys())[0]
    best_model = models[best_model_name]
    predicted_fares = best_model.predict(X_test.iloc[sample_indices])

    plt.scatter(distances, predicted_fares, alpha=0.3, s=20,
                label=f'{best_model_name} Predictions', c='red')

    plt.xlabel('Distance (miles)', fontsize=12)
    plt.ylabel('Fare ($)', fontsize=12)
    plt.title('Fare vs Distance: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 30])
    plt.ylim([0, 100])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fare_vs_distance.png', dpi=300, bbox_inches='tight')
    print(f"Fare vs distance plot saved to {output_dir}/fare_vs_distance.png")
    plt.close()


def save_best_model(models, results_df, output_dir='models'):
    """
    Save the best performing model (lowest RMSE)
    """
    os.makedirs(output_dir, exist_ok=True)

    best_model_name = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
    best_model = models[best_model_name]

    model_path = f'{output_dir}/best_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"\nBest model ({best_model_name}) saved to {model_path}")

    return best_model_name


def main():
    """
    Main training pipeline
    """
    print("\n=== NYC Taxi Fare Prediction - Model Training ===\n")

    df = pd.read_csv('data/featured_data.csv')
    print(f"Loaded {len(df):,} records with {df.shape[1]} features\n")

    X, y = prepare_features(df)

    print("Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}\n")

    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    os.makedirs('results', exist_ok=True)
    results.to_csv('results/model_metrics.csv', index=False)
    print("✅ Metrics saved to results/model_metrics.csv\n")

    plot_predictions(models, X_test, y_test)

    # Use best tree-based model for feature importance
    best_tree_model = models.get('Random Forest') or models.get('Gradient Boosting')
    if best_tree_model:
        plot_feature_importance(best_tree_model, X.columns.tolist())

    plot_fare_vs_distance(X_test, y_test, models)

    save_best_model(models, results)

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()