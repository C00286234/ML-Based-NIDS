"""
NIDS Randomised Evaluation
===========================
Runs attack detection tests N times with different random seeds and larger
samples to produce mean accuracy +/- std with 95% confidence intervals.

This addresses the supervisor's concern about randomisation: a single fixed-seed
test run doesn't prove the system generalises — multiple randomised runs do.

Author: C00286234

Usage:
    python randomised_evaluation.py                        # 10 runs x 20 samples/class
    python randomised_evaluation.py --runs 5 --samples 10  # Quick test
    python randomised_evaluation.py --runs 20 --samples 30 # Thorough evaluation
    python randomised_evaluation.py --url http://host:8000  # Custom API URL
"""

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from src.constants import FEATURE_NAMES_84

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_API_URL = "http://localhost:8000"
DATA_PATH = Path(__file__).parent / "data" / "processed" / "cicids2017_cleaned.csv"

# Formatting
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

BATCH_CHUNK_SIZE = 1000  # Max flows per batch_predict call


def print_header(text):
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_dataset(data_path):
    """Load and clean the full dataset once."""
    print(f"\n  Loading dataset from {data_path}...")

    if not data_path.exists():
        print(f"\n{RED}ERROR: Dataset not found at {data_path}{RESET}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df = df.replace([float('inf'), float('-inf')], 0).fillna(0)
    print(f"  Loaded {len(df):,} flows ({df['Label'].nunique()} classes)")

    # Show class distribution
    class_counts = df["Label"].value_counts()
    for label, count in class_counts.items():
        print(f"    {label:<35} {count:>8,}")

    return df


def sample_for_run(df, samples_per_class, seed):
    """Sample flows for a single evaluation run."""
    sampled = []
    for label, group in df.groupby("Label"):
        n = min(samples_per_class, len(group))
        sampled.append(group.sample(n=n, random_state=seed))
    return pd.concat(sampled, ignore_index=True)


# ==============================================================================
# BATCH PREDICTION
# ==============================================================================

def predict_batch(api_url, flows_df):
    """Send flows via /batch_predict, chunking if needed. Returns list of predictions."""
    all_predictions = []

    for chunk_start in range(0, len(flows_df), BATCH_CHUNK_SIZE):
        chunk = flows_df.iloc[chunk_start:chunk_start + BATCH_CHUNK_SIZE]

        # Build batch payload (ordered list format)
        flows = []
        for _, row in chunk.iterrows():
            features = [float(row[name]) for name in FEATURE_NAMES_84]
            flows.append({"features": features})

        payload = {"flows": flows}

        resp = requests.post(f"{api_url}/batch_predict", json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        all_predictions.extend(data.get("predictions", []))

    return all_predictions


# ==============================================================================
# EVALUATION LOGIC
# ==============================================================================

def run_single_evaluation(api_url, df, samples_per_class, seed):
    """
    Run one evaluation round: sample data, predict, score.

    Returns:
        per_class: dict of {label: {"correct": n, "total": n}}
        confusions: list of (true_label, predicted_label) for wrong predictions
    """
    sample = sample_for_run(df, samples_per_class, seed)
    true_labels = sample["Label"].tolist()

    predictions = predict_batch(api_url, sample)

    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    confusions = []

    for true_label, pred_data in zip(true_labels, predictions):
        predicted = pred_data["prediction"]
        per_class[true_label]["total"] += 1

        if predicted == true_label:
            per_class[true_label]["correct"] += 1
        else:
            confusions.append((true_label, predicted))

    return dict(per_class), confusions


def run_all_evaluations(api_url, df, samples_per_class, n_runs, base_seed):
    """Run N evaluation rounds and aggregate results."""
    print_header(f"Running {n_runs} Evaluation Rounds")

    all_run_results = []      # list of per_class dicts
    all_confusions = []       # aggregated (true, pred) pairs
    all_overall_accuracies = []  # overall accuracy per run
    per_class_accuracies = defaultdict(list)  # label -> [acc1, acc2, ...]

    # Track sample sizes per class
    class_counts = df["Label"].value_counts().to_dict()

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        print(f"\n  Run {run_idx + 1}/{n_runs} (seed={seed})...", end=" ", flush=True)

        start = time.perf_counter()
        per_class, confusions = run_single_evaluation(
            api_url, df, samples_per_class, seed
        )
        elapsed = time.perf_counter() - start

        all_run_results.append(per_class)
        all_confusions.extend(confusions)

        # Calculate overall accuracy for this run
        total_correct = sum(v["correct"] for v in per_class.values())
        total_tested = sum(v["total"] for v in per_class.values())
        overall_acc = total_correct / total_tested * 100 if total_tested > 0 else 0
        all_overall_accuracies.append(overall_acc)

        # Per-class accuracy for this run
        for label, counts in per_class.items():
            acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
            per_class_accuracies[label].append(acc)

        n_wrong = len(confusions) - sum(
            len(c) for c in [confusions]  # just count this run's confusions
        ) + len(confusions)
        print(f"{overall_acc:.1f}% ({total_correct}/{total_tested}) in {elapsed:.1f}s")

    return (all_run_results, all_confusions, all_overall_accuracies,
            per_class_accuracies, class_counts)


# ==============================================================================
# REPORTING
# ==============================================================================

def print_and_save_report(all_overall_accuracies, per_class_accuracies,
                          all_confusions, class_counts, samples_per_class,
                          n_runs, timestamp):
    """Print report and save to files."""

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # --- Aggregate stats ---
    overall_mean = statistics.mean(all_overall_accuracies)
    overall_std = statistics.stdev(all_overall_accuracies) if len(all_overall_accuracies) > 1 else 0
    ci_margin = 1.96 * overall_std / (len(all_overall_accuracies) ** 0.5)
    ci_low = overall_mean - ci_margin
    ci_high = overall_mean + ci_margin

    # Confusion patterns
    confusion_counts = defaultdict(int)
    for true_label, predicted in all_confusions:
        confusion_counts[(true_label, predicted)] += 1
    top_confusions = sorted(confusion_counts.items(), key=lambda x: -x[1])[:10]

    # --- Print to console ---
    print_header("Randomised Evaluation Results")

    color = GREEN if overall_mean >= 90 else YELLOW if overall_mean >= 70 else RED
    print(f"\n  {BOLD}Overall Accuracy: {color}{overall_mean:.1f}% +/- {overall_std:.1f}%{RESET}"
          f"  (95% CI: [{ci_low:.1f}%, {ci_high:.1f}%])")
    print(f"  Runs: {n_runs}  |  Samples/class: {samples_per_class}")

    # Per-class table
    print(f"\n  {'Attack Type':<35} {'Mean Acc':>10} {'Std':>8} {'N/run':>8}")
    print(f"  {'-'*65}")

    for label in sorted(per_class_accuracies.keys()):
        accs = per_class_accuracies[label]
        mean_acc = statistics.mean(accs)
        std_acc = statistics.stdev(accs) if len(accs) > 1 else 0
        dataset_size = class_counts.get(label, 0)
        actual_n = min(samples_per_class, dataset_size)
        capped = "*" if dataset_size < samples_per_class else ""

        color = GREEN if mean_acc >= 90 else YELLOW if mean_acc >= 70 else RED
        print(f"  {label:<35} {color}{mean_acc:>9.1f}%{RESET} {std_acc:>7.1f}% {actual_n:>6}{capped}")

    if top_confusions:
        print(f"\n  {BOLD}Top Confusion Patterns:{RESET}")
        for (true_label, predicted), count in top_confusions:
            print(f"    {true_label} -> {predicted:<30} ({count} times)")

    # Note about capped classes
    capped_classes = [label for label, count in class_counts.items()
                      if count < samples_per_class]
    if capped_classes:
        print(f"\n  {YELLOW}* = class has fewer than {samples_per_class} samples in dataset; "
              f"capped at dataset size{RESET}")

    # --- Save text report ---
    report_path = results_dir / f"randomised_eval_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("NIDS Randomised Evaluation Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Runs:           {n_runs}\n")
        f.write(f"Samples/class:  {samples_per_class}\n\n")

        f.write(f"Overall Accuracy: {overall_mean:.1f}% +/- {overall_std:.1f}%\n")
        f.write(f"95% CI:           [{ci_low:.1f}%, {ci_high:.1f}%]\n\n")

        f.write(f"{'Attack Type':<35} {'Mean Acc':>10} {'Std':>8} {'N/run':>8}\n")
        f.write(f"{'-'*65}\n")

        for label in sorted(per_class_accuracies.keys()):
            accs = per_class_accuracies[label]
            mean_acc = statistics.mean(accs)
            std_acc = statistics.stdev(accs) if len(accs) > 1 else 0
            dataset_size = class_counts.get(label, 0)
            actual_n = min(samples_per_class, dataset_size)
            capped = "*" if dataset_size < samples_per_class else ""
            f.write(f"{label:<35} {mean_acc:>9.1f}% {std_acc:>7.1f}% {actual_n:>6}{capped}\n")

        if top_confusions:
            f.write(f"\nTop Confusion Patterns:\n")
            for (true_label, predicted), count in top_confusions:
                f.write(f"  {true_label} -> {predicted:<30} ({count} times)\n")

        f.write(f"\nPer-run overall accuracy:\n")
        for i, acc in enumerate(all_overall_accuracies):
            f.write(f"  Run {i+1}: {acc:.1f}%\n")

    print(f"\n  {GREEN}Report saved:{RESET}  {report_path}")

    # --- Save CSV ---
    csv_path = results_dir / f"randomised_eval_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "label", "correct", "total", "accuracy"])

        # We need per-run per-class data — reconstruct from per_class_accuracies
        # Since we stored accuracies in order, and we also need correct/total,
        # we actually need the raw run results. Let's use the stored data.
        # For simplicity, write what we have:
        for label in sorted(per_class_accuracies.keys()):
            for run_idx, acc in enumerate(per_class_accuracies[label]):
                # Approximate correct/total from accuracy
                dataset_size = class_counts.get(label, 0)
                actual_n = min(samples_per_class, dataset_size)
                correct = round(acc * actual_n / 100)
                writer.writerow([run_idx + 1, label, correct, actual_n, f"{acc:.1f}"])

    print(f"  {GREEN}CSV saved:{RESET}     {csv_path}")
    return report_path, csv_path


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="NIDS Randomised Evaluation")
    parser.add_argument("--url", default=DEFAULT_API_URL,
                        help=f"API URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--data", default=str(DATA_PATH),
                        help=f"Path to cleaned CSV (default: {DATA_PATH})")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of evaluation runs (default: 10)")
    parser.add_argument("--samples", type=int, default=20,
                        help="Samples per class per run (default: 20)")
    parser.add_argument("--seed", type=int, default=100,
                        help="Base random seed (default: 100)")
    args = parser.parse_args()

    print_header("NIDS RANDOMISED EVALUATION")
    print(f"  API:            {args.url}")
    print(f"  Data:           {args.data}")
    print(f"  Runs:           {args.runs}")
    print(f"  Samples/class:  {args.samples}")
    print(f"  Base seed:      {args.seed}")

    # Check API health
    try:
        resp = requests.get(f"{args.url}/health", timeout=5)
        health = resp.json()
        print(f"  API Status:     {GREEN}Online{RESET} (uptime: {health.get('uptime_seconds', 0):.0f}s)")
        if not health.get("models_loaded"):
            print(f"  {RED}ERROR: Models not loaded!{RESET}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"\n{RED}ERROR: API not reachable at {args.url}{RESET}")
        print(f"  Start it with: python api/app.py")
        sys.exit(1)

    # Load data once
    df = load_dataset(Path(args.data))

    # Run evaluations
    (all_run_results, all_confusions, all_overall_accuracies,
     per_class_accuracies, class_counts) = run_all_evaluations(
        args.url, df, args.samples, args.runs, args.seed
    )

    # Report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print_and_save_report(
        all_overall_accuracies, per_class_accuracies,
        all_confusions, class_counts, args.samples,
        args.runs, timestamp
    )

    print(f"\n{BOLD}Evaluation complete.{RESET}\n")


if __name__ == "__main__":
    main()
