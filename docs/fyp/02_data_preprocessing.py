# ============================================================================
# STEP 2: DATA PREPROCESSING & CLEANING
# Multi-class Classification for Network Intrusion Detection
# ============================================================================
# Converts raw CIC-IDS2017 CSVs into a single cleaned dataset.
#
# Input:  CICIDS2017_improved/*.csv  (5 day files)
# Output: data/processed/cicids2017_cleaned.csv
#         figures/label_distribution.png
#
# Usage:  python 02_data_preprocessing.py
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for scripts
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(os.path.expanduser("~"), "nids_extras", "CICIDS2017_improved")
PROCESSED_PATH = os.path.join(BASE_PATH, "data", "processed")
FIGURES_PATH = os.path.join(BASE_PATH, "figures")

CSV_FILES = ['monday.csv', 'tuesday.csv', 'wednesday.csv', 'thursday.csv', 'friday.csv']

# Columns to drop (not useful for ML)
COLS_TO_DROP = ['id', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Attempted Category']


def setup_directories():
    """Create required output directories."""
    for folder in [PROCESSED_PATH, FIGURES_PATH]:
        os.makedirs(folder, exist_ok=True)
        print(f"  [OK] {os.path.relpath(folder, BASE_PATH)}")


def load_and_combine(data_path, csv_files):
    """Load all 5 days of CIC-IDS2017 data and combine into one DataFrame."""

    print("\n" + "=" * 70)
    print("LOADING ALL 5 DAYS OF DATA")
    print("=" * 70)

    dataframes = []
    for filename in csv_files:
        filepath = os.path.join(data_path, filename)
        if not os.path.exists(filepath):
            print(f"  [WARN] {filename} not found, skipping")
            continue
        print(f"  Loading {filename}...", end="")
        df = pd.read_csv(filepath)
        dataframes.append(df)
        print(f" {len(df):,} flows")

    if not dataframes:
        print("[ERROR] No CSV files found. Check CICIDS2017_improved/ folder.")
        sys.exit(1)

    print("\nCombining all days...")
    df_combined = pd.concat(dataframes, ignore_index=True)
    print(f"  Combined dataset: {len(df_combined):,} flows, {df_combined.shape[1]} columns")
    print(f"  Memory usage: {df_combined.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

    del dataframes  # Free memory
    return df_combined


def consolidate_labels(df):
    """Remove ' - Attempted' suffix from attack labels."""

    print("\n" + "=" * 70)
    print("CONSOLIDATING 'ATTEMPTED' LABELS")
    print("=" * 70)

    before = df['Label'].nunique()
    df['Label'] = df['Label'].str.replace(' - Attempted', '', regex=False)
    after = df['Label'].nunique()

    print(f"  Classes before: {before} -> after: {after}")
    print(f"\n  Label distribution:")
    for label, count in df['Label'].value_counts().items():
        print(f"    {label:40s}: {count:>10,}")

    return df


def visualize_distribution(df):
    """Save class distribution bar chart."""

    print("\n" + "=" * 70)
    print("VISUALIZING CLASS DISTRIBUTION")
    print("=" * 70)

    plt.figure(figsize=(14, 6))
    label_counts = df['Label'].value_counts()
    label_counts.plot(kind='bar')
    plt.title('Distribution of Attack Types (After Consolidation)', fontsize=14)
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_PATH, 'label_distribution.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved to {os.path.relpath(fig_path, BASE_PATH)}")


def clean_problematic_values(df):
    """Handle infinity values and suspicious negatives."""

    print("\n" + "=" * 70)
    print("CLEANING PROBLEMATIC VALUES")
    print("=" * 70)

    feature_cols = [col for col in df.columns if col != 'Label']

    # --- Infinity values ---
    inf_count_total = 0
    for col in feature_cols:
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            inf_mask = np.isinf(df[col])
            inf_count = inf_mask.sum()
            if inf_count > 0:
                inf_count_total += inf_count
                # Replace inf with NaN, then fill with median
                df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill all NaN with column median
    for col in feature_cols:
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

    print(f"  [OK] Replaced {inf_count_total:,} infinity values with median")

    # --- Negative header lengths ---
    header_cols = ['Fwd Header Length', 'Bwd Header Length']
    for col in header_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                df[col] = df[col].abs()
                print(f"  [OK] Fixed {neg_count:,} negative values in {col}")

    return df


def remove_non_feature_columns(df):
    """Drop columns that aren't useful for ML."""

    print("\n" + "=" * 70)
    print("REMOVING NON-FEATURE COLUMNS")
    print("=" * 70)

    existing_cols_to_drop = [col for col in COLS_TO_DROP if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        print(f"  Dropped: {existing_cols_to_drop}")
    else:
        print("  No columns to drop")

    print(f"  Remaining: {df.shape[1]} columns ({df.shape[1] - 1} features + Label)")
    return df


def save_processed(df):
    """Save cleaned dataset to CSV."""

    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATA")
    print("=" * 70)

    output_path = os.path.join(PROCESSED_PATH, "cicids2017_cleaned.csv")
    df.to_csv(output_path, index=False)

    file_size = os.path.getsize(output_path) / 1024**2
    print(f"  [OK] Saved {len(df):,} flows to {os.path.relpath(output_path, BASE_PATH)}")
    print(f"       File size: {file_size:.2f} MB")

    return output_path


def main():
    print("=" * 70)
    print("STEP 2: DATA PREPROCESSING & CLEANING")
    print("=" * 70)

    # Setup
    print("\nChecking directories...")
    setup_directories()

    # Load
    df = load_and_combine(RAW_DATA_PATH, CSV_FILES)

    # Clean labels
    df = consolidate_labels(df)

    # Visualize
    visualize_distribution(df)

    # Clean values
    df = clean_problematic_values(df)

    # Drop non-feature columns
    df = remove_non_feature_columns(df)

    # Save
    output_path = save_processed(df)

    # Summary
    missing = df.isnull().sum().sum()
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Total flows:    {len(df):,}")
    print(f"  Total features: {df.shape[1] - 1}")
    print(f"  Total classes:  {df['Label'].nunique()}")
    print(f"  Missing values: {missing}")
    print(f"  Output:         {os.path.relpath(output_path, BASE_PATH)}")
    print(f"\n  Next step: python 04_model_training_improved.py")


if __name__ == "__main__":
    main()
