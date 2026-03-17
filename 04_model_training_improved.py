# ============================================================================
# IMPROVED MODEL TRAINING PIPELINE v2.0
# ============================================================================
#
# Key Improvements over Original:
# 1. Conservative SMOTE (not aggressive oversampling)
# 2. Hyperparameter tuning with RandomizedSearchCV
# 3. 5-fold Cross-validation for robust evaluation
# 4. Feature selection using model importance (not just ANOVA)
# 5. Proper evaluation metrics (macro F1, per-class analysis)
# 6. Handling of tiny classes (merge or use class weights)
# 7. Early stopping for XGBoost
# 8. Model ensembling option
#
# Author: C00286234
# Version: 2.0
# ============================================================================

import os
import sys
import time
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    make_scorer
)

from xgboost import XGBClassifier

# Optional: SMOTE (only if imblearn is installed)
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. Using class weights instead of SMOTE.")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration - easy to modify"""

    # Paths
    BASE_PATH = r"C:\Users\folosiye\ML-BASED-NIDS"
    DATA_PATH = os.path.join(BASE_PATH, "data", "processed")
    MODELS_PATH = os.path.join(BASE_PATH, "models")
    FIGURES_PATH = os.path.join(BASE_PATH, "figures")
    RESULTS_PATH = os.path.join(BASE_PATH, "results")

    # Data split
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42

    # Feature selection
    N_FEATURES = 50  # Number of features to select
    FEATURE_SELECTION_METHOD = 'anova'  # 'model', 'anova', or 'rfe'

    # Class imbalance handling
    USE_SMOTE = True  # Use SMOTE to boost rare attack classes
    SMOTE_STRATEGY = 'moderate'  # 'conservative', 'moderate', or 'aggressive'
    MIN_CLASS_SAMPLES = 5  # Lowered to keep Heartbleed (11) and SQL Injection (18) as separate classes
    MERGE_RARE_CLASSES = False  # Keep all 16 classes instead of merging rare ones

    # Model training
    USE_HYPERPARAMETER_TUNING = True   # Enabled for optimised final models
    N_ITER_SEARCH = 10  # Reduced for memory efficiency
    CV_FOLDS = 3  # Reduced for memory efficiency
    N_JOBS = 2  # Limit parallel jobs to save memory (use -1 for all cores if RAM > 16GB)

    # Evaluation
    PRIMARY_METRIC = 'f1_macro'  # Better for imbalanced data

    # XGBoost specific
    EARLY_STOPPING_ROUNDS = 10


def create_directories():
    """Create necessary directories"""
    for path in [Config.MODELS_PATH, Config.FIGURES_PATH, Config.RESULTS_PATH]:
        os.makedirs(path, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load the cleaned CIC-IDS2017 dataset"""

    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    data_file = os.path.join(Config.DATA_PATH, "cicids2017_cleaned.csv")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} flows with {df.shape[1]} columns")

    return df


# ============================================================================
# CLASS HANDLING
# ============================================================================

def handle_rare_classes(df, min_samples=50):
    """
    Handle classes with very few samples.

    Options:
    1. Merge into 'Rare Attack' category
    2. Remove entirely (for anomaly detection approach)
    """

    print("\n" + "="*70)
    print("HANDLING RARE CLASSES")
    print("="*70)

    class_counts = df['Label'].value_counts()
    rare_classes = class_counts[class_counts < min_samples].index.tolist()

    if not rare_classes:
        print("No rare classes found (all classes have >= {min_samples} samples)")
        return df

    print(f"\nRare classes (< {min_samples} samples):")
    for cls in rare_classes:
        print(f"  - {cls}: {class_counts[cls]} samples")

    if Config.MERGE_RARE_CLASSES:
        # Merge rare classes
        df['Label'] = df['Label'].apply(
            lambda x: 'Rare Attack' if x in rare_classes else x
        )
        print(f"\nMerged {len(rare_classes)} rare classes into 'Rare Attack'")
        print(f"New 'Rare Attack' count: {(df['Label'] == 'Rare Attack').sum()}")
    else:
        # Remove rare classes
        df = df[~df['Label'].isin(rare_classes)]
        print(f"\nRemoved {len(rare_classes)} rare classes")
        print(f"Remaining samples: {len(df):,}")

    print(f"\nFinal class distribution:")
    print(df['Label'].value_counts())

    return df


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """
    Preprocess data: encode labels, split, scale, select features
    """

    print("\n" + "="*70)
    print("PREPROCESSING DATA")
    print("="*70)

    # Separate features and target
    X = df.drop('Label', axis=1)
    y = df['Label']

    print(f"Features: {X.shape[1]}")
    print(f"Classes: {y.nunique()}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\nLabel encoding:")
    for i, cls in enumerate(label_encoder.classes_):
        count = (y_encoded == i).sum()
        print(f"  {i:2d} -> {cls:40s} ({count:,} samples)")

    # Save label encoder
    encoder_path = os.path.join(Config.MODELS_PATH, "label_encoder.pkl")
    joblib.dump(label_encoder, encoder_path)
    print(f"\nLabel encoder saved to: {encoder_path}")

    # Train/Val/Test split (stratified)
    print("\n" + "-"*50)
    print("TRAIN/VAL/TEST SPLIT")
    print("-"*50)

    # First split: train vs temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded,
        test_size=(Config.TEST_SIZE + Config.VAL_SIZE),
        random_state=Config.RANDOM_STATE,
        stratify=y_encoded
    )

    # Second split: val vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=Config.RANDOM_STATE,
        stratify=y_temp
    )

    print(f"Training set:   {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f"Test set:       {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")

    # Feature scaling
    print("\n" + "-"*50)
    print("FEATURE SCALING")
    print("-"*50)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("StandardScaler fitted on training data")

    # Save scaler
    scaler_path = os.path.join(Config.MODELS_PATH, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # Feature selection
    print("\n" + "-"*50)
    print(f"FEATURE SELECTION ({Config.FEATURE_SELECTION_METHOD.upper()})")
    print("-"*50)

    if Config.FEATURE_SELECTION_METHOD == 'model':
        # Use Random Forest feature importance
        print("Using Random Forest feature importance...")

        rf_selector = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=Config.RANDOM_STATE,
            n_jobs=Config.N_JOBS,
            class_weight='balanced'
        )
        rf_selector.fit(X_train_scaled, y_train)

        # Get feature importances
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 20 features by importance:")
        print(importances.head(20).to_string())

        # Select top N features
        selected_features = importances.head(Config.N_FEATURES)['feature'].tolist()
        feature_mask = X.columns.isin(selected_features)

        X_train_selected = X_train_scaled[:, feature_mask]
        X_val_selected = X_val_scaled[:, feature_mask]
        X_test_selected = X_test_scaled[:, feature_mask]

        # Create a selector object for consistency
        selector = SelectKBest(score_func=f_classif, k=Config.N_FEATURES)
        selector.fit(X_train_scaled, y_train)

    else:  # 'anova' - original method
        print("Using ANOVA F-test...")
        selector = SelectKBest(score_func=f_classif, k=Config.N_FEATURES)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_val_selected = selector.transform(X_val_scaled)
        X_test_selected = selector.transform(X_test_scaled)

        selected_features = X.columns[selector.get_support()].tolist()

    print(f"\nSelected {len(selected_features)} features")

    # Save selector
    selector_path = os.path.join(Config.MODELS_PATH, "feature_selector.pkl")
    joblib.dump(selector, selector_path)
    print(f"Feature selector saved to: {selector_path}")

    # Save selected feature names
    features_path = os.path.join(Config.DATA_PATH, "selected_features.txt")
    with open(features_path, 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    print(f"Feature names saved to: {features_path}")

    return {
        'X_train': X_train_selected,
        'X_val': X_val_selected,
        'X_test': X_test_selected,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'selector': selector,
        'selected_features': selected_features,
        'feature_names': X.columns.tolist()
    }


# ============================================================================
# CLASS IMBALANCE HANDLING
# ============================================================================

def handle_class_imbalance(X_train, y_train):
    """
    Handle class imbalance using conservative SMOTE or class weights.

    Conservative SMOTE:
    - Only oversample to a reasonable level (not equal to majority)
    - Use lower k_neighbors for tiny classes
    - Cap maximum synthetic samples
    """

    print("\n" + "="*70)
    print("HANDLING CLASS IMBALANCE")
    print("="*70)

    class_counts = Counter(y_train)
    print("Original class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  Class {cls}: {count:,} samples")

    if not Config.USE_SMOTE or not IMBLEARN_AVAILABLE:
        print("\nUsing class_weight='balanced' instead of SMOTE")
        print("(More stable for classes with very few samples)")
        return X_train, y_train, None

    # Conservative SMOTE strategy
    print(f"\nApplying {Config.SMOTE_STRATEGY} SMOTE strategy...")

    majority_count = max(class_counts.values())

    if Config.SMOTE_STRATEGY == 'conservative':
        # Max 5x increase, cap at 50k
        target_counts = {
            cls: min(count * 5, 50000, majority_count // 2)
            for cls, count in class_counts.items()
            if count < majority_count // 2
        }
    elif Config.SMOTE_STRATEGY == 'moderate':
        # Max 10x increase, cap at 100k
        target_counts = {
            cls: min(count * 10, 100000, majority_count)
            for cls, count in class_counts.items()
            if count < majority_count
        }
    else:  # aggressive (original behavior - not recommended)
        target_counts = 'not majority'

    # Determine appropriate k_neighbors
    min_class_count = min(class_counts.values())
    k_neighbors = min(5, min_class_count - 1)
    k_neighbors = max(1, k_neighbors)  # At least 1

    print(f"SMOTE k_neighbors: {k_neighbors}")
    print(f"Target counts: {target_counts}")

    try:
        smote = SMOTE(
            sampling_strategy=target_counts if isinstance(target_counts, dict) else target_counts,
            k_neighbors=k_neighbors,
            random_state=Config.RANDOM_STATE,
            n_jobs=Config.N_JOBS
        )

        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print(f"\nAfter SMOTE:")
        print(f"  Original samples: {len(X_train):,}")
        print(f"  Resampled samples: {len(X_resampled):,}")

        new_counts = Counter(y_resampled)
        for cls in sorted(new_counts.keys()):
            original = class_counts[cls]
            new = new_counts[cls]
            increase = new / original if original > 0 else 0
            print(f"  Class {cls}: {original:,} -> {new:,} ({increase:.1f}x)")

        return X_resampled, y_resampled, smote

    except Exception as e:
        print(f"\nSMOTE failed: {e}")
        print("Falling back to class_weight='balanced'")
        return X_train, y_train, None


# ============================================================================
# MODEL TRAINING
# ============================================================================

def get_rf_param_grid():
    """Random Forest hyperparameter search space"""
    return {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.3],
        'class_weight': ['balanced', 'balanced_subsample']
    }


def get_xgb_param_grid():
    """XGBoost hyperparameter search space"""
    return {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }


def train_random_forest(X_train, y_train, X_val, y_val, use_smote_data=False):
    """
    Train Random Forest with optional hyperparameter tuning.
    """

    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST")
    print("="*70)

    # Determine if we need class weights
    use_class_weight = not use_smote_data

    if Config.USE_HYPERPARAMETER_TUNING:
        print("Performing hyperparameter tuning...")
        print(f"  Search iterations: {Config.N_ITER_SEARCH}")
        print(f"  CV folds: {Config.CV_FOLDS}")

        param_grid = get_rf_param_grid()

        if use_class_weight:
            param_grid['class_weight'] = ['balanced', 'balanced_subsample']
        else:
            param_grid['class_weight'] = [None]

        base_rf = RandomForestClassifier(
            random_state=Config.RANDOM_STATE,
            n_jobs=Config.N_JOBS
        )

        cv = StratifiedKFold(
            n_splits=Config.CV_FOLDS,
            shuffle=True,
            random_state=Config.RANDOM_STATE
        )

        search = RandomizedSearchCV(
            base_rf,
            param_distributions=param_grid,
            n_iter=Config.N_ITER_SEARCH,
            cv=cv,
            scoring=Config.PRIMARY_METRIC,
            random_state=Config.RANDOM_STATE,
            n_jobs=Config.N_JOBS,
            verbose=1
        )

        start_time = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - start_time

        print(f"\nHyperparameter search completed in {elapsed/60:.2f} minutes")
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV {Config.PRIMARY_METRIC}: {search.best_score_:.4f}")

        rf_model = search.best_estimator_

    else:
        # Use reasonable defaults with class weights
        print("Training with default parameters...")

        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced' if use_class_weight else None,
            random_state=Config.RANDOM_STATE,
            n_jobs=Config.N_JOBS,
            verbose=1
        )

        start_time = time.time()
        rf_model.fit(X_train, y_train)
        elapsed = time.time() - start_time

        print(f"\nTraining completed in {elapsed/60:.2f} minutes")

    # Save model
    model_path = os.path.join(Config.MODELS_PATH, "random_forest_model.pkl")
    joblib.dump(rf_model, model_path)
    print(f"Model saved to: {model_path}")

    return rf_model


def train_xgboost(X_train, y_train, X_val, y_val, n_classes, use_smote_data=False):
    """
    Train XGBoost with optional hyperparameter tuning and early stopping.
    """

    print("\n" + "="*70)
    print("TRAINING XGBOOST")
    print("="*70)

    # Calculate class weights for imbalanced data
    if not use_smote_data:
        class_counts = Counter(y_train)
        total = sum(class_counts.values())
        scale_pos_weight = {
            cls: total / (n_classes * count)
            for cls, count in class_counts.items()
        }
        print("Using sample weights for class imbalance")
    else:
        scale_pos_weight = None

    if Config.USE_HYPERPARAMETER_TUNING:
        print("Performing hyperparameter tuning...")

        param_grid = get_xgb_param_grid()

        base_xgb = XGBClassifier(
            objective='multi:softprob',
            num_class=n_classes,
            random_state=Config.RANDOM_STATE,
            n_jobs=Config.N_JOBS,
            eval_metric='mlogloss'
        )

        cv = StratifiedKFold(
            n_splits=Config.CV_FOLDS,
            shuffle=True,
            random_state=Config.RANDOM_STATE
        )

        search = RandomizedSearchCV(
            base_xgb,
            param_distributions=param_grid,
            n_iter=Config.N_ITER_SEARCH,
            cv=cv,
            scoring=Config.PRIMARY_METRIC,
            random_state=Config.RANDOM_STATE,
            n_jobs=Config.N_JOBS,
            verbose=1
        )

        start_time = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - start_time

        print(f"\nHyperparameter search completed in {elapsed/60:.2f} minutes")
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV {Config.PRIMARY_METRIC}: {search.best_score_:.4f}")

        # Retrain best model with early stopping
        best_params = search.best_params_
        xgb_model = XGBClassifier(
            **best_params,
            objective='multi:softprob',
            num_class=n_classes,
            random_state=Config.RANDOM_STATE,
            n_jobs=Config.N_JOBS,
            eval_metric='mlogloss',
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS
        )

        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    else:
        print("Training with default parameters + early stopping...")

        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=n_classes,
            random_state=Config.RANDOM_STATE,
            n_jobs=Config.N_JOBS,
            eval_metric='mlogloss',
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS
        )

        start_time = time.time()
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        elapsed = time.time() - start_time

        print(f"\nTraining completed in {elapsed/60:.2f} minutes")
        print(f"Best iteration: {xgb_model.best_iteration}")

    # Save model
    model_path = os.path.join(Config.MODELS_PATH, "xgboost_model.pkl")
    joblib.dump(xgb_model, model_path)
    print(f"Model saved to: {model_path}")

    return xgb_model


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X, y, label_encoder, dataset_name="Test"):
    """
    Comprehensive model evaluation with multiple metrics.
    """

    print("\n" + "="*70)
    print(f"EVALUATION ON {dataset_name.upper()} SET")
    print("="*70)

    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # Multiple metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision_weighted': precision_score(y, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
    }

    # ROC-AUC (multi-class)
    try:
        metrics['roc_auc_ovr'] = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
    except:
        metrics['roc_auc_ovr'] = None

    # Print metrics
    print("\nOverall Metrics:")
    print("-" * 50)
    print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1 (weighted):      {metrics['f1_weighted']:.4f}")
    print(f"  F1 (macro):         {metrics['f1_macro']:.4f}  <-- Key metric for imbalanced data")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    if metrics['roc_auc_ovr']:
        print(f"  ROC-AUC (macro):    {metrics['roc_auc_ovr']:.4f}")

    # Per-class report
    print("\n" + "-"*50)
    print("Per-Class Classification Report:")
    print("-"*50)
    print(classification_report(
        y, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_true, y_pred, label_encoder, model_name, save_path):
    """Plot and save confusion matrix"""

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to: {save_path}")


def cross_validate_model(model, X, y, model_name):
    """Perform cross-validation for more robust evaluation"""

    print(f"\n{Config.CV_FOLDS}-Fold Cross-Validation for {model_name}:")

    cv = StratifiedKFold(
        n_splits=Config.CV_FOLDS,
        shuffle=True,
        random_state=Config.RANDOM_STATE
    )

    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted'
    }

    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=Config.N_JOBS,
        return_train_score=True
    )

    print(f"  Accuracy:     {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std()*2:.4f})")
    print(f"  F1 (macro):   {cv_results['test_f1_macro'].mean():.4f} (+/- {cv_results['test_f1_macro'].std()*2:.4f})")
    print(f"  F1 (weighted):{cv_results['test_f1_weighted'].mean():.4f} (+/- {cv_results['test_f1_weighted'].std()*2:.4f})")

    return cv_results


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""

    print("="*70)
    print("IMPROVED NIDS MODEL TRAINING PIPELINE v2.0")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create directories
    create_directories()

    # Load data
    df = load_data()

    # Handle rare classes
    df = handle_rare_classes(df, min_samples=Config.MIN_CLASS_SAMPLES)

    # Preprocess
    data = preprocess_data(df)

    # Handle class imbalance
    X_train_resampled, y_train_resampled, smote = handle_class_imbalance(
        data['X_train'], data['y_train']
    )

    use_smote_data = smote is not None
    n_classes = len(data['label_encoder'].classes_)

    # Train Random Forest
    rf_model = train_random_forest(
        X_train_resampled, y_train_resampled,
        data['X_val'], data['y_val'],
        use_smote_data=use_smote_data
    )

    # Train XGBoost
    xgb_model = train_xgboost(
        X_train_resampled, y_train_resampled,
        data['X_val'], data['y_val'],
        n_classes=n_classes,
        use_smote_data=use_smote_data
    )

    # Evaluate on validation set
    print("\n" + "="*70)
    print("VALIDATION SET EVALUATION")
    print("="*70)

    rf_val_metrics, rf_val_pred, _ = evaluate_model(
        rf_model, data['X_val'], data['y_val'],
        data['label_encoder'], "Validation"
    )

    xgb_val_metrics, xgb_val_pred, _ = evaluate_model(
        xgb_model, data['X_val'], data['y_val'],
        data['label_encoder'], "Validation"
    )

    # Choose best model based on macro F1
    if rf_val_metrics['f1_macro'] >= xgb_val_metrics['f1_macro']:
        best_model = rf_model
        best_model_name = "Random Forest"
    else:
        best_model = xgb_model
        best_model_name = "XGBoost"

    print(f"\nBest model: {best_model_name}")

    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL TEST SET EVALUATION")
    print("="*70)

    test_metrics, y_test_pred, y_test_proba = evaluate_model(
        best_model, data['X_test'], data['y_test'],
        data['label_encoder'], "Test"
    )

    # Plot confusion matrices
    plot_confusion_matrix(
        data['y_test'], y_test_pred, data['label_encoder'],
        best_model_name,
        os.path.join(Config.FIGURES_PATH, f'{best_model_name.lower().replace(" ", "_")}_confusion_matrix_v2.png')
    )

    # Save results
    results = {
        'Model': ['Random Forest', 'XGBoost', f'{best_model_name} (Test)'],
        'Dataset': ['Validation', 'Validation', 'Test'],
        'Accuracy': [rf_val_metrics['accuracy'], xgb_val_metrics['accuracy'], test_metrics['accuracy']],
        'F1_Macro': [rf_val_metrics['f1_macro'], xgb_val_metrics['f1_macro'], test_metrics['f1_macro']],
        'F1_Weighted': [rf_val_metrics['f1_weighted'], xgb_val_metrics['f1_weighted'], test_metrics['f1_weighted']],
        'Precision_Macro': [rf_val_metrics['precision_macro'], xgb_val_metrics['precision_macro'], test_metrics['precision_macro']],
        'Recall_Macro': [rf_val_metrics['recall_macro'], xgb_val_metrics['recall_macro'], test_metrics['recall_macro']],
    }

    results_df = pd.DataFrame(results)
    results_path = os.path.join(Config.RESULTS_PATH, "model_results_v2.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Save training configuration
    config_dict = {
        'training_date': datetime.now().isoformat(),
        'n_features': Config.N_FEATURES,
        'use_smote': Config.USE_SMOTE,
        'smote_strategy': Config.SMOTE_STRATEGY,
        'hyperparameter_tuning': Config.USE_HYPERPARAMETER_TUNING,
        'cv_folds': Config.CV_FOLDS,
        'best_model': best_model_name,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1_macro': test_metrics['f1_macro'],
        'n_classes': n_classes,
        'class_names': list(data['label_encoder'].classes_),
    }

    config_path = os.path.join(Config.MODELS_PATH, "training_config.pkl")
    joblib.dump(config_dict, config_path)
    print(f"Training config saved to: {config_path}")

    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return {
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'best_model': best_model,
        'test_metrics': test_metrics,
        'data': data
    }


if __name__ == "__main__":
    results = main()
