"""
Kali Test Results Parser
=========================
Parses NIDS session logs against a test plan (from run_kali_tests.py) to
determine which attacks were detected, missed, or misclassified.

Author: C00286234

Reads predictions.log from the NIDS session directory and correlates
each prediction's timestamp against the attack time windows from the test plan.

Usage:
    python testing/parse_kali_results.py --plan results/kali_test_plan_YYYYMMDD.json
    python testing/parse_kali_results.py --plan results/kali_test_plan_YYYYMMDD.json --session logs/sessions/YYYYMMDD_HHMMSS
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ==============================================================================
# FORMATTING
# ==============================================================================

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

RESULTS_DIR = Path(__file__).parent.parent / "results"


def print_header(text):
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")


# ==============================================================================
# LOG PARSING
# ==============================================================================

# Log format: "2026-03-05 16:03:45 | ATTACK | DoS Hulk                     | conf=0.9842 | 192.168.1.5:443 -> 192.168.1.10:54321 (TCP) | pkts=150"
LOG_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*"  # timestamp
    r"(ATTACK|BENIGN)\s*\|\s*"                             # type
    r"(.+?)\s*\|\s*"                                       # label (with trailing spaces)
    r"conf=(\d+\.\d+)\s*\|\s*"                            # confidence
    r"(.+?)\s*\|\s*"                                       # flow summary (src:port -> dst:port (proto))
    r"pkts=(\d+)"                                          # packet count
)


def parse_predictions_log(log_path):
    """Parse a predictions.log file into structured records."""
    predictions = []

    if not log_path.exists():
        print(f"  {RED}predictions.log not found at {log_path}{RESET}")
        return predictions

    with open(log_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            match = LOG_PATTERN.match(line)
            if match:
                timestamp_str, attack_type, label, confidence, flow_summary, pkts = match.groups()
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

                predictions.append({
                    "timestamp": timestamp,
                    "is_attack": attack_type == "ATTACK",
                    "predicted_label": label.strip(),
                    "confidence": float(confidence),
                    "flow_summary": flow_summary.strip(),
                    "packets": int(pkts),
                    "line_num": line_num,
                })
            else:
                # Try to handle minor format variations
                if "|" in line and "conf=" in line:
                    print(f"  {YELLOW}Warning: Could not parse line {line_num}: {line[:80]}...{RESET}")

    return predictions


def find_session_dir(test_plan):
    """Find the most likely session directory for a test plan."""
    session_start = datetime.fromisoformat(test_plan["session_start"])
    sessions_root = Path(__file__).parent.parent / "logs" / "sessions"

    if not sessions_root.exists():
        return None

    # Look for session directories that started around the same time
    best_match = None
    best_diff = float("inf")

    for session_dir in sessions_root.iterdir():
        if not session_dir.is_dir():
            continue
        try:
            dir_time = datetime.strptime(session_dir.name, "%Y%m%d_%H%M%S")
            diff = abs((session_start - dir_time).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_match = session_dir
        except ValueError:
            continue

    if best_match and best_diff < 3600:  # Within 1 hour
        return best_match

    return None


# ==============================================================================
# CORRELATION
# ==============================================================================

def correlate_predictions(predictions, test_plan):
    """Match predictions to test time windows."""
    tests = test_plan["tests"]
    results = []

    for test in tests:
        test_start = datetime.fromisoformat(test["start_time"])
        test_end = datetime.fromisoformat(test["end_time"])
        expected_class = test["expected_class"]

        # Find all predictions within this time window
        # Include a buffer of 130s after the window end (flow timeout)
        from datetime import timedelta
        buffer_end = test_end + timedelta(seconds=130)

        matched = []
        for pred in predictions:
            if test_start <= pred["timestamp"] <= buffer_end:
                matched.append(pred)

        # Classify each prediction
        correct = 0
        attack_detected = 0
        benign_count = 0
        wrong_class = 0
        pred_labels = defaultdict(int)

        for pred in matched:
            pred_labels[pred["predicted_label"]] += 1

            if pred["predicted_label"] == expected_class:
                correct += 1
                attack_detected += 1
            elif pred["is_attack"]:
                attack_detected += 1
                wrong_class += 1
            else:
                benign_count += 1

        total = len(matched)

        # Detection rate: any attack detected (even wrong class) vs totally missed
        detection_rate = attack_detected / total * 100 if total > 0 else 0
        # Exact match rate: correct class specifically
        exact_rate = correct / total * 100 if total > 0 else 0
        # Confidence for correct predictions
        correct_confidences = [
            p["confidence"] for p in matched
            if p["predicted_label"] == expected_class
        ]

        result = {
            "test_id": test["test_id"],
            "test_name": test["name"],
            "expected_class": expected_class,
            "total_flows": total,
            "correct": correct,
            "wrong_class": wrong_class,
            "benign_misclass": benign_count,
            "detection_rate": detection_rate,
            "exact_match_rate": exact_rate,
            "prediction_distribution": dict(pred_labels),
            "avg_confidence": statistics.mean(correct_confidences) if correct_confidences else 0,
            "matched_predictions": matched,
        }
        results.append(result)

    return results


# ==============================================================================
# REPORTING
# ==============================================================================

def print_report(results, test_plan):
    """Print a colored report to console."""
    print_header("Kali Live Attack Test Results")

    session_start = test_plan["session_start"][:19]
    print(f"  Session: {session_start}")
    print(f"  Tests:   {len(results)}")

    # Per-test results
    total_flows = 0
    total_detected = 0
    total_exact = 0

    print(f"\n  {'#':>3} {'Attack':30} {'Flows':>7} {'Detected':>10} {'Exact':>10} {'Top Prediction':>30}")
    print(f"  {'-'*95}")

    for r in results:
        total_flows += r["total_flows"]
        total_detected += r["correct"] + r["wrong_class"]
        total_exact += r["correct"]

        # Color based on detection
        if r["exact_match_rate"] >= 70:
            color = GREEN
        elif r["detection_rate"] >= 50:
            color = YELLOW
        else:
            color = RED

        # Top prediction
        if r["prediction_distribution"]:
            top_pred = max(r["prediction_distribution"], key=r["prediction_distribution"].get)
            top_count = r["prediction_distribution"][top_pred]
            top_str = f"{top_pred} ({top_count})"
        else:
            top_str = "No flows"

        print(f"  {r['test_id']:>3} {r['test_name']:30} {r['total_flows']:>7} "
              f"{color}{r['detection_rate']:>9.0f}%{RESET} "
              f"{color}{r['exact_match_rate']:>9.0f}%{RESET} "
              f"{top_str:>30}")

    # Overall
    overall_detection = total_detected / total_flows * 100 if total_flows > 0 else 0
    overall_exact = total_exact / total_flows * 100 if total_flows > 0 else 0

    print(f"\n  {BOLD}Overall:{RESET}")
    print(f"    Total flows processed: {total_flows}")
    print(f"    Attack detected (any): {overall_detection:.1f}%")
    print(f"    Exact class match:     {overall_exact:.1f}%")

    # Detailed breakdown for each test
    print_header("Detailed Breakdown")

    for r in results:
        if r["total_flows"] == 0:
            print(f"\n  {BOLD}#{r['test_id']} {r['test_name']}{RESET}")
            print(f"    {RED}No flows captured during this test window!{RESET}")
            print(f"    Possible causes:")
            print(f"      - Attack didn't generate IP traffic visible to the NIDS")
            print(f"      - Wrong interface being captured")
            print(f"      - Flows still pending (increase wait time)")
            continue

        print(f"\n  {BOLD}#{r['test_id']} {r['test_name']}{RESET}")
        print(f"    Expected:   {r['expected_class']}")
        print(f"    Flows:      {r['total_flows']}")
        print(f"    Detection:  {r['detection_rate']:.0f}% (any attack class)")
        print(f"    Exact:      {r['exact_match_rate']:.0f}% (correct class)")
        if r['avg_confidence'] > 0:
            print(f"    Avg conf:   {r['avg_confidence']:.2%}")

        if r["prediction_distribution"]:
            print(f"    Predictions:")
            for label, count in sorted(r["prediction_distribution"].items(),
                                       key=lambda x: -x[1]):
                pct = count / r["total_flows"] * 100
                marker = f" {GREEN}<< correct{RESET}" if label == r["expected_class"] else ""
                print(f"      {label:<35} {count:>5} ({pct:.0f}%){marker}")


def save_report(results, test_plan, timestamp):
    """Save report to text and CSV files."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Text report ---
    report_path = RESULTS_DIR / f"kali_live_test_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("NIDS Kali Live Attack Test Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Session: {test_plan['session_start'][:19]}\n")
        f.write(f"Tests:   {len(results)}\n\n")

        f.write(f"{'#':>3} {'Attack':30} {'Flows':>7} {'Detected':>10} {'Exact':>10}\n")
        f.write(f"{'-'*65}\n")

        total_flows = 0
        total_exact = 0

        for r in results:
            total_flows += r["total_flows"]
            total_exact += r["correct"]
            f.write(f"{r['test_id']:>3} {r['test_name']:30} {r['total_flows']:>7} "
                    f"{r['detection_rate']:>9.0f}% {r['exact_match_rate']:>9.0f}%\n")

            if r["prediction_distribution"]:
                for label, count in sorted(r["prediction_distribution"].items(),
                                           key=lambda x: -x[1]):
                    f.write(f"      -> {label}: {count}\n")

        overall = total_exact / total_flows * 100 if total_flows > 0 else 0
        f.write(f"\nOverall exact match: {overall:.1f}% ({total_exact}/{total_flows})\n")

    print(f"\n  {GREEN}Report saved:{RESET} {report_path}")

    # --- CSV with per-flow predictions ---
    csv_path = RESULTS_DIR / f"kali_live_test_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_id", "test_name", "expected_class", "timestamp",
            "predicted_label", "confidence", "is_attack", "correct",
            "flow_summary", "packets"
        ])

        for r in results:
            for pred in r["matched_predictions"]:
                writer.writerow([
                    r["test_id"],
                    r["test_name"],
                    r["expected_class"],
                    pred["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    pred["predicted_label"],
                    f"{pred['confidence']:.4f}",
                    pred["is_attack"],
                    pred["predicted_label"] == r["expected_class"],
                    pred["flow_summary"],
                    pred["packets"],
                ])

    print(f"  {GREEN}CSV saved:{RESET}    {csv_path}")
    return report_path, csv_path


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parse NIDS session logs against Kali attack test plan"
    )
    parser.add_argument("--plan", required=True,
                        help="Path to test plan JSON (from run_kali_tests.py)")
    parser.add_argument("--session", default=None,
                        help="Path to NIDS session directory (auto-detected if not specified)")
    args = parser.parse_args()

    # Load test plan
    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"{RED}Test plan not found: {plan_path}{RESET}")
        sys.exit(1)

    with open(plan_path, "r", encoding="utf-8") as f:
        test_plan = json.load(f)

    print_header("Kali Test Results Parser")
    print(f"  Test plan:  {plan_path}")
    print(f"  Session:    {test_plan['session_start'][:19]}")
    print(f"  Tests:      {test_plan['total_tests']}")

    # Find session directory
    if args.session:
        session_dir = Path(args.session)
    else:
        session_dir = find_session_dir(test_plan)
        if session_dir:
            print(f"  Auto-detected session: {session_dir}")
        else:
            print(f"  {RED}Could not auto-detect session directory.{RESET}")
            print(f"  Use --session <path> to specify manually.")

            # List available sessions
            sessions_root = Path(__file__).parent.parent / "logs" / "sessions"
            if sessions_root.exists():
                dirs = sorted(sessions_root.iterdir(), reverse=True)
                print(f"\n  Recent sessions:")
                for d in dirs[:5]:
                    print(f"    {d}")
            sys.exit(1)

    # Parse predictions log
    predictions_log = session_dir / "predictions.log"
    print(f"  Parsing:    {predictions_log}")

    predictions = parse_predictions_log(predictions_log)
    print(f"  Found {len(predictions)} predictions in log")

    if not predictions:
        print(f"\n  {RED}No predictions found! Check that the NIDS engine was running.{RESET}")
        sys.exit(1)

    # Correlate predictions with test windows
    results = correlate_predictions(predictions, test_plan)

    # Print report
    print_report(results, test_plan)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_report(results, test_plan, timestamp)


if __name__ == "__main__":
    main()
