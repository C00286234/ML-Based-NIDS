"""
NIDS Attack Detection Test Script
==================================
Loads real attack samples from CIC-IDS2017 test data and sends them to the
NIDS API to validate that the system correctly detects each attack type.

Author: C00286234

Usage:
    python test_attack_detection.py                     # Test all attack types (5 samples each)
    python test_attack_detection.py --attack DDoS        # Test a specific attack type
    python test_attack_detection.py --samples 20         # More samples per type
    python test_attack_detection.py --batch              # Use batch prediction endpoint
    python test_attack_detection.py --url http://host:8000  # Custom API URL
"""

import argparse
import csv
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

# ==============================================================================
# FORMATTING
# ==============================================================================

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(text):
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")


def print_result(label, prediction, confidence, is_attack, correct):
    status = f"{GREEN}CORRECT{RESET}" if correct else f"{RED}WRONG{RESET}"
    attack_indicator = f"{RED}ATTACK{RESET}" if is_attack else f"{GREEN}BENIGN{RESET}"
    print(f"  [{status}] True: {label:<30} Predicted: {prediction:<30} "
          f"Confidence: {confidence:.2%}  {attack_indicator}")


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_attack_samples(data_path, attack_type=None, samples_per_type=5):
    """Load attack samples from the CIC-IDS2017 dataset."""

    print(f"\n  Loading dataset from {data_path}...")

    if not data_path.exists():
        print(f"\n{RED}ERROR: Dataset not found at {data_path}{RESET}")
        print(f"  Run notebooks 01-03 to preprocess the CIC-IDS2017 data first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} flows ({df['Label'].nunique()} classes)")

    # Replace inf/-inf with 0 and fill NaN
    df = df.replace([float('inf'), float('-inf')], 0).fillna(0)

    # Get attack types
    attack_types = df[df["Label"] != "BENIGN"]["Label"].unique()

    if attack_type:
        # Filter to specific attack type
        matches = [t for t in attack_types if attack_type.lower() in t.lower()]
        if not matches:
            print(f"\n{RED}ERROR: Attack type '{attack_type}' not found.{RESET}")
            print(f"  Available types: {', '.join(sorted(attack_types))}")
            sys.exit(1)
        attack_types = matches

    # Sample from each attack type + BENIGN
    samples = []

    # Always include some benign samples for comparison
    benign = df[df["Label"] == "BENIGN"].sample(n=min(samples_per_type, len(df[df["Label"] == "BENIGN"])), random_state=42)
    samples.append(benign)

    for atype in sorted(attack_types):
        subset = df[df["Label"] == atype]
        n = min(samples_per_type, len(subset))
        sample = subset.sample(n=n, random_state=42)
        samples.append(sample)
        print(f"  Sampled {n} flows for: {atype}")

    result = pd.concat(samples, ignore_index=True)
    print(f"  Total test samples: {len(result)}")
    return result


# ==============================================================================
# API TESTING
# ==============================================================================

def test_single_predictions(df, api_url):
    """Send each sample to /predict one at a time."""

    print_header("Single Prediction Tests")

    results = defaultdict(lambda: {"correct": 0, "total": 0, "predictions": []})
    total_correct = 0
    total_tested = 0
    latencies = []
    prediction_log = []  # Store every prediction for CSV export

    for label, group in df.groupby("Label"):
        print(f"\n  --- {BOLD}{label}{RESET} ({len(group)} samples) ---")

        for _, row in group.iterrows():
            features = {name: float(row[name]) for name in FEATURE_NAMES_84}
            payload = {"features": features}

            try:
                start = time.time()
                resp = requests.post(f"{api_url}/predict", json=payload, timeout=30)
                latency = (time.time() - start) * 1000
                latencies.append(latency)

                if resp.status_code != 200:
                    print(f"  {RED}ERROR: HTTP {resp.status_code} — {resp.text[:100]}{RESET}")
                    continue

                data = resp.json()
                prediction = data["prediction"]
                confidence = data["confidence"]
                is_attack = data["is_attack"]

                # Check correctness
                correct = prediction == label
                total_tested += 1
                results[label]["total"] += 1

                if correct:
                    total_correct += 1
                    results[label]["correct"] += 1

                results[label]["predictions"].append(prediction)
                print_result(label, prediction, confidence, is_attack, correct)

                # Log for CSV export
                prediction_log.append({
                    "true_label": label,
                    "predicted_label": prediction,
                    "confidence": round(confidence, 4),
                    "is_attack": is_attack,
                    "correct": correct,
                    "latency_ms": round(latency, 1),
                })

            except requests.exceptions.ConnectionError:
                print(f"\n{RED}ERROR: Cannot connect to API at {api_url}{RESET}")
                print(f"  Make sure the API is running: cd api && python -m uvicorn app:app --port 8000")
                sys.exit(1)
            except Exception as e:
                print(f"  {RED}ERROR: {e}{RESET}")

    return results, total_correct, total_tested, latencies, prediction_log


def test_batch_predictions(df, api_url):
    """Send all samples in one batch to /batch_predict."""

    print_header("Batch Prediction Test")

    # Build batch payload
    flows = []
    labels = []
    for _, row in df.iterrows():
        features = {name: float(row[name]) for name in FEATURE_NAMES_84}
        flows.append({"features": features})
        labels.append(row["Label"])

    payload = {"flows": flows}
    print(f"  Sending {len(flows)} flows to /batch_predict...")

    try:
        start = time.time()
        resp = requests.post(f"{api_url}/batch_predict", json=payload, timeout=120)
        latency = (time.time() - start) * 1000

        if resp.status_code != 200:
            print(f"  {RED}ERROR: HTTP {resp.status_code} — {resp.text[:200]}{RESET}")
            return None

        data = resp.json()
        predictions = data.get("predictions", [])
        print(f"  Received {len(predictions)} predictions in {latency:.0f}ms "
              f"({latency/len(predictions):.1f}ms per flow)")

        # Evaluate
        results = defaultdict(lambda: {"correct": 0, "total": 0})
        total_correct = 0

        for label, pred_data in zip(labels, predictions):
            prediction = pred_data["prediction"]
            correct = prediction == label
            results[label]["total"] += 1
            if correct:
                results[label]["correct"] += 1
                total_correct += 1

        return results, total_correct, len(labels), latency

    except requests.exceptions.ConnectionError:
        print(f"\n{RED}ERROR: Cannot connect to API at {api_url}{RESET}")
        sys.exit(1)


# ==============================================================================
# REPORTING
# ==============================================================================

def print_summary(results, total_correct, total_tested, latencies=None):
    """Print a summary of detection results."""

    print_header("Detection Summary")

    overall_accuracy = total_correct / total_tested * 100 if total_tested > 0 else 0
    color = GREEN if overall_accuracy >= 90 else YELLOW if overall_accuracy >= 70 else RED

    print(f"\n  {BOLD}Overall Accuracy: {color}{overall_accuracy:.1f}%{RESET} "
          f"({total_correct}/{total_tested})")

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"  {BOLD}Avg Latency:{RESET} {avg_latency:.0f}ms  |  {BOLD}P95:{RESET} {p95:.0f}ms")

    # Per-class breakdown
    print(f"\n  {'Attack Type':<35} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*65}")

    for label in sorted(results.keys()):
        r = results[label]
        acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
        color = GREEN if acc >= 90 else YELLOW if acc >= 70 else RED
        is_benign = " (benign)" if label == "BENIGN" else ""
        print(f"  {label + is_benign:<35} {r['correct']:>8} {r['total']:>8} {color}{acc:>9.1f}%{RESET}")

    # Highlight misclassifications
    misclassified = {k: v for k, v in results.items() if v.get("predictions")}
    wrong_predictions = []
    for label, data in misclassified.items():
        for pred in data.get("predictions", []):
            if pred != label:
                wrong_predictions.append((label, pred))

    if wrong_predictions:
        print(f"\n  {BOLD}{YELLOW}Misclassifications:{RESET}")
        for true_label, predicted in wrong_predictions:
            print(f"    {true_label} → predicted as {predicted}")

    return overall_accuracy


# ==============================================================================
# RESULT SAVING
# ==============================================================================

def save_results(results, total_correct, total_tested, prediction_log, latencies=None):
    """Save test results to CSV and summary text file in results/."""

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Save prediction log CSV ---
    csv_path = results_dir / f"attack_test_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "true_label", "predicted_label", "confidence", "is_attack", "correct", "latency_ms"
        ])
        writer.writeheader()
        writer.writerows(prediction_log)

    print(f"\n  {GREEN}Prediction log saved:{RESET}  {csv_path}")

    # --- Save summary text ---
    summary_path = results_dir / f"attack_test_summary_{timestamp}.txt"
    overall_accuracy = total_correct / total_tested * 100 if total_tested > 0 else 0

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("NIDS Attack Detection Test Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Date:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Overall Accuracy:  {overall_accuracy:.1f}% ({total_correct}/{total_tested})\n")

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p95 = sorted(latencies)[int(len(latencies) * 0.95)]
            f.write(f"Avg Latency:       {avg_latency:.0f}ms\n")
            f.write(f"P95 Latency:       {p95:.0f}ms\n")

        f.write(f"\n{'Attack Type':<35} {'Correct':>8} {'Total':>8} {'Accuracy':>10}\n")
        f.write(f"{'-'*65}\n")

        for label in sorted(results.keys()):
            r = results[label]
            acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
            tag = " (benign)" if label == "BENIGN" else ""
            f.write(f"{label + tag:<35} {r['correct']:>8} {r['total']:>8} {acc:>9.1f}%\n")

        # Misclassifications
        wrong = []
        for label, data in results.items():
            for pred in data.get("predictions", []):
                if pred != label:
                    wrong.append((label, pred))
        if wrong:
            f.write(f"\nMisclassifications:\n")
            for true_label, predicted in wrong:
                f.write(f"  {true_label} -> predicted as {predicted}\n")

    print(f"  {GREEN}Summary saved:{RESET}       {summary_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test NIDS attack detection with real CIC-IDS2017 data"
    )
    parser.add_argument("--url", default=DEFAULT_API_URL,
                        help=f"API URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--attack", default=None,
                        help="Test a specific attack type (e.g., 'DDoS', 'Portscan')")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of samples per attack type (default: 5)")
    parser.add_argument("--batch", action="store_true",
                        help="Use /batch_predict instead of /predict")
    parser.add_argument("--data", default=str(DATA_PATH),
                        help=f"Path to cleaned CSV (default: {DATA_PATH})")
    args = parser.parse_args()

    print_header("NIDS Attack Detection Test")
    print(f"  API:     {args.url}")
    print(f"  Data:    {args.data}")
    print(f"  Samples: {args.samples} per attack type")
    print(f"  Mode:    {'Batch' if args.batch else 'Single'} prediction")

    # Check API is up
    try:
        resp = requests.get(f"{args.url}/health", timeout=5)
        health = resp.json()
        print(f"  API:     {GREEN}Online{RESET} (uptime: {health.get('uptime_seconds', 0):.0f}s)")
        if not health.get("models_loaded"):
            print(f"  {RED}WARNING: Models not loaded! Predictions will fail.{RESET}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"\n{RED}ERROR: API not reachable at {args.url}{RESET}")
        print(f"  Start it with: cd api && python -m uvicorn app:app --port 8000")
        sys.exit(1)

    # Load data
    df = load_attack_samples(Path(args.data), args.attack, args.samples)

    # Run tests
    prediction_log = []
    if args.batch:
        result = test_batch_predictions(df, args.url)
        if result:
            results, total_correct, total_tested, latency = result
            print_summary(results, total_correct, total_tested)
            print(f"\n  Batch latency: {latency:.0f}ms total ({latency/total_tested:.1f}ms per flow)")
            save_results(results, total_correct, total_tested, prediction_log)
    else:
        results, total_correct, total_tested, latencies, prediction_log = test_single_predictions(df, args.url)
        print_summary(results, total_correct, total_tested, latencies)
        save_results(results, total_correct, total_tested, prediction_log, latencies)

    print()


if __name__ == "__main__":
    main()
