"""
NYC Taxi Fare Prediction - Complete Pipeline
Runs preprocessing, feature engineering, and model training
"""

import sys
import os


def main():
    """Execute the complete ML pipeline"""
    print("="*60)
    print("NYC TAXI FARE PREDICTION - COMPLETE PIPELINE")
    print("="*60)

    data_path = 'data/train.csv'
    if not os.path.exists(data_path):
        print(f"\n❌ Error: Data file not found at {data_path}")
        print("\nPlease download the Kaggle dataset:")
        print("1. Visit: https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data")
        print("2. Download train.csv (5.7 GB)")
        print("3. Place it in data/train.csv")
        sys.exit(1)

    try:
        from src.data_preprocessing import load_data, preprocess_data
        from src.feature_engineering import engineer_features
        from src.model_training import (
            prepare_features, train_models, evaluate_models,
            plot_predictions, plot_feature_importance, plot_fare_vs_distance,
            save_best_model
        )
        from sklearn.model_selection import train_test_split

        # STEP 1: Data Preprocessing
        print("\n" + "="*60)
        print("STEP 1: DATA PREPROCESSING")
        print("="*60)

        # HIGH ACCURACY: Use 500k-2M rows (adjust based on RAM)
        df = load_data(data_path, nrows=1000000)
        df_clean = preprocess_data(df)

        df_clean.to_csv('data/preprocessed_data.csv', index=False)
        print("✅ Preprocessed data saved to data/preprocessed_data.csv")

        # STEP 2: Feature Engineering
        print("\n" + "="*60)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*60)

        df_features = engineer_features(df_clean)

        df_features.to_csv('data/featured_data.csv', index=False)
        print("✅ Featured data saved to data/featured_data.csv")

        # STEP 3: Model Training
        print("\n" + "="*60)
        print("STEP 3: MODEL TRAINING & EVALUATION")
        print("="*60)

        X, y = prepare_features(df_features)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")

        print("\nTraining models (this may take 25-40 minutes)...")
        models = train_models(X_train, y_train)

        results = evaluate_models(models, X_test, y_test)

        os.makedirs('results', exist_ok=True)
        results.to_csv('results/model_metrics.csv', index=False)
        print("✅ Metrics saved to results/model_metrics.csv")

        plot_predictions(models, X_test, y_test)

        # Use best tree-based model for feature importance
        best_tree_model = models.get('Random Forest') or models.get('Gradient Boosting')
        if best_tree_model:
            plot_feature_importance(best_tree_model, X.columns.tolist())

        plot_fare_vs_distance(X_test, y_test, models)

        best_model_name = save_best_model(models, results)

        # Final Summary
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETE!")
        print("="*60)
        print("\nGenerated Files:")
        print("  📁 data/preprocessed_data.csv - Cleaned data")
        print("  📁 data/featured_data.csv - Data with engineered features")
        print("  📁 models/best_model.pkl - Best trained model")
        print("  📁 results/model_metrics.csv - Model performance metrics")
        print("  📁 results/model_predictions.png - Prediction visualizations")
        print("  📁 results/feature_importance.png - Feature importance plot")
        print("  📁 results/fare_vs_distance.png - Fare vs distance analysis")
        print(f"\n🏆 Best Model: {best_model_name}")
        print("\nYou can now use the trained model for predictions!")
        print("Run: python3 predict.py")

    except ImportError as e:
        print(f"\n❌ Import Error: {str(e)}")
        print("\nMake sure all dependencies are installed:")
        print("  pip3 install -r requirements.txt")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()