# ============================================================================
# STEP 3: FEATURE ENGINEERING & TRAIN-TEST SPLIT
# Prepare data for multi-class classification (16 attack types)
# Author: C00286234
# ============================================================================
# Loads the cleaned CSV, encodes labels, splits data, scales features,
# selects top 50 features, and applies SMOTE for class balancing.
#
# Input:  data/processed/cicids2017_cleaned.csv
# Output: models/label_encoder.pkl
#         models/scaler.pkl
#         models/feature_selector.pkl
#         data/processed/X_train_resampled.npy
#         data/processed/y_train_resampled.npy
#         data/processed/X_val.npy, y_val.npy
#         data/processed/X_test.npy, y_test.npy
#         data/processed/selected_features.txt
#
# Usage:  python 03_feature_engineering.py
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data", "processed")
MODELS_PATH = os.path.join(BASE_PATH, "models")

# Split ratios
TEST_SIZE = 0.30   # 30% held out (then split 50/50 into val/test)
RANDOM_STATE = 42
N_FEATURES = 50    # Top features to select via ANOVA F-test


def setup_directories():
    """Create required output directories."""
    for folder in [DATA_PATH, MODELS_PATH]:
        os.makedirs(folder, exist_ok=True)
        print(f"  [OK] {os.path.relpath(folder, BASE_PATH)}")


def load_cleaned_data():
    """Load the cleaned CSV from step 02."""

    print("\n" + "=" * 70)
    print("LOADING CLEANED DATA")
    print("=" * 70)

    data_path = os.path.join(DATA_PATH, "cicids2017_cleaned.csv")
    if not os.path.exists(data_path):
        print(f"[ERROR] File not found: {data_path}")
        print("        Run 02_data_preprocessing.py first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} flows, {df.shape[1]} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def separate_features_target(df):
    """Split DataFrame into features (X) and target (y)."""

    print("\n" + "=" * 70)
    print("SEPARATING FEATURES AND TARGET")
    print("=" * 70)

    y = df['Label']
    X = df.drop('Label', axis=1)

    print(f"  Target: {y.name}")
    print(f"  Classes: {y.nunique()}")
    print(f"  Features: {X.shape[1]}")

    print(f"\n  Class distribution:")
    for label, count in y.value_counts().items():
        print(f"    {label:40s}: {count:>10,}")

    return X, y


def encode_labels(y):
    """Convert string labels to integer codes."""

    print("\n" + "=" * 70)
    print("ENCODING TARGET LABELS")
    print("=" * 70)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("  Label encoding mapping:")
    for i, label in enumerate(label_encoder.classes_):
        count = (y_encoded == i).sum()
        print(f"    {i:2d} -> {label:40s} ({count:>10,} flows)")

    y_encoded = pd.Series(y_encoded, name='Label_Encoded')
    print(f"\n  Total classes: {len(label_encoder.classes_)}")

    # Save
    encoder_path = os.path.join(MODELS_PATH, "label_encoder.pkl")
    joblib.dump(label_encoder, encoder_path)
    print(f"  [OK] Saved to {os.path.relpath(encoder_path, BASE_PATH)}")

    return y_encoded, label_encoder


def split_data(X, y_encoded):
    """Stratified 70/15/15 train/val/test split."""

    print("\n" + "=" * 70)
    print("TRAIN-TEST SPLIT (STRATIFIED)")
    print("=" * 70)

    # First: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    # Second: split temp 50/50 into val and test (each 15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    total = len(X)
    print(f"  Training set:   {len(X_train):,} flows ({len(X_train)/total*100:.1f}%)")
    print(f"  Validation set: {len(X_val):,} flows ({len(X_val)/total*100:.1f}%)")
    print(f"  Test set:       {len(X_test):,} flows ({len(X_test)/total*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test, feature_columns):
    """Fit StandardScaler on training data, transform all sets."""

    print("\n" + "=" * 70)
    print("FEATURE SCALING (STANDARDIZATION)")
    print("=" * 70)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns)

    print(f"  Mean of training data: {X_train_scaled.mean().mean():.6f} (should be ~0)")
    print(f"  Std of training data:  {X_train_scaled.std().mean():.6f} (should be ~1)")

    # Save
    scaler_path = os.path.join(MODELS_PATH, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  [OK] Saved to {os.path.relpath(scaler_path, BASE_PATH)}")

    return scaler, X_train_scaled, X_val_scaled, X_test_scaled


def select_features(X_train_scaled, X_val_scaled, X_test_scaled, y_train, feature_columns):
    """Select top N features using ANOVA F-test."""

    print("\n" + "=" * 70)
    print(f"FEATURE SELECTION (TOP {N_FEATURES} FEATURES)")
    print("=" * 70)

    selector = SelectKBest(score_func=f_classif, k=N_FEATURES)
    selector.fit(X_train_scaled, y_train)

    # Show top features
    feature_scores = pd.DataFrame({
        'Feature': feature_columns,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)

    print("\n  Top 20 most important features:")
    for _, row in feature_scores.head(20).iterrows():
        print(f"    {row['Feature']:40s}: {row['Score']:>15,.2f}")

    # Apply selection
    X_train_selected = selector.transform(X_train_scaled)
    X_val_selected = selector.transform(X_val_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    selected_features = feature_columns[selector.get_support()].tolist()
    print(f"\n  Selected {len(selected_features)} features")

    # Convert to DataFrames
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
    X_val_selected = pd.DataFrame(X_val_selected, columns=selected_features)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)

    # Save selector
    selector_path = os.path.join(MODELS_PATH, "feature_selector.pkl")
    joblib.dump(selector, selector_path)
    print(f"  [OK] Saved to {os.path.relpath(selector_path, BASE_PATH)}")

    return selector, X_train_selected, X_val_selected, X_test_selected, selected_features


def apply_smote(X_train_selected, y_train):
    """Apply SMOTE oversampling to balance classes in training data only."""

    print("\n" + "=" * 70)
    print("HANDLING CLASS IMBALANCE WITH SMOTE")
    print("=" * 70)

    print("  Class distribution BEFORE SMOTE:")
    for cls, count in pd.Series(y_train).value_counts().sort_index().items():
        print(f"    Class {cls:2d}: {count:>10,}")

    print("\n  Applying SMOTE (this may take a few minutes)...")

    smote = SMOTE(
        sampling_strategy='not majority',
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )

    X_resampled, y_resampled = smote.fit_resample(X_train_selected, y_train)

    print(f"\n  Training set BEFORE SMOTE: {len(X_train_selected):,}")
    print(f"  Training set AFTER SMOTE:  {len(X_resampled):,}")

    print("\n  Class distribution AFTER SMOTE:")
    for cls, count in pd.Series(y_resampled).value_counts().sort_index().items():
        print(f"    Class {cls:2d}: {count:>10,}")

    return X_resampled, y_resampled


def save_all(X_train_resampled, y_train_resampled,
             X_val_selected, y_val,
             X_test_selected, y_test,
             selected_features):
    """Save all processed data as numpy arrays."""

    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATA")
    print("=" * 70)

    # Training data (SMOTE applied)
    np.save(os.path.join(DATA_PATH, "X_train_resampled.npy"), X_train_resampled)
    np.save(os.path.join(DATA_PATH, "y_train_resampled.npy"), y_train_resampled)
    print("  [OK] Saved resampled training data")

    # Validation data (original)
    np.save(os.path.join(DATA_PATH, "X_val.npy"), X_val_selected)
    np.save(os.path.join(DATA_PATH, "y_val.npy"), y_val)
    print("  [OK] Saved validation data")

    # Test data (original)
    np.save(os.path.join(DATA_PATH, "X_test.npy"), X_test_selected)
    np.save(os.path.join(DATA_PATH, "y_test.npy"), y_test)
    print("  [OK] Saved test data")

    # Feature names
    features_path = os.path.join(DATA_PATH, "selected_features.txt")
    with open(features_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    print("  [OK] Saved feature names")


def main():
    print("=" * 70)
    print("STEP 3: FEATURE ENGINEERING & TRAIN-TEST SPLIT")
    print("=" * 70)

    # Setup
    print("\nChecking directories...")
    setup_directories()

    # Load
    df = load_cleaned_data()

    # Separate
    X, y = separate_features_target(df)
    feature_columns = X.columns

    # Encode labels
    y_encoded, label_encoder = encode_labels(y)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_encoded)

    # Scale
    scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_features(
        X_train, X_val, X_test, feature_columns
    )

    # Select features
    selector, X_train_sel, X_val_sel, X_test_sel, selected_features = select_features(
        X_train_scaled, X_val_scaled, X_test_scaled, y_train, feature_columns
    )

    # SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train_sel, y_train)

    # Save
    save_all(
        X_train_resampled, y_train_resampled,
        X_val_sel, y_val,
        X_test_sel, y_test,
        selected_features
    )

    # Summary
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print(f"  Original dataset:   {len(df):,} flows, {X.shape[1]} features")
    print(f"  Selected features:  {len(selected_features)}")
    print(f"  Number of classes:  {len(label_encoder.classes_)}")
    print(f"  Training set:       {len(X_train_resampled):,} flows (SMOTE applied)")
    print(f"  Validation set:     {len(X_val_sel):,} flows (original)")
    print(f"  Test set:           {len(X_test_sel):,} flows (original)")
    print(f"\n  Saved artifacts:")
    print(f"    models/label_encoder.pkl")
    print(f"    models/scaler.pkl")
    print(f"    models/feature_selector.pkl")
    print(f"    data/processed/X_train_resampled.npy")
    print(f"    data/processed/y_train_resampled.npy")
    print(f"    data/processed/X_val.npy, y_val.npy")
    print(f"    data/processed/X_test.npy, y_test.npy")
    print(f"    data/processed/selected_features.txt")
    print(f"\n  Next step: python 04_model_training_improved.py")


if __name__ == "__main__":
    main()
