"""
NIDS Benchmark Suite
====================
Measures system performance: latency percentiles, batch throughput,
concurrent throughput, and single-vs-batch speedup.

Usage:
    python benchmark_suite.py                          # Default (skip concurrent)
    python benchmark_suite.py --skip-concurrent        # Safe without disabling rate limit
    python benchmark_suite.py --single-n 100           # More single-prediction samples
    python benchmark_suite.py --url http://host:8000   # Custom API URL
"""

import argparse
import csv
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from nids.features import FEATURE_NAMES_84
from nids.paths import PROCESSED_DIR, REPORTS_DIR

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_API_URL = "http://localhost:8000"
DATA_PATH = PROCESSED_DIR / "test.parquet"  # held-out flows only

# Formatting
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


def print_metric(name, value, unit=""):
    print(f"    {name:<30} {value:>10} {unit}")


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_benchmark_data(data_path, n_samples, seed):
    """Load stratified random samples from the dataset."""
    print(f"\n  Loading dataset from {data_path}...")

    if not data_path.exists():
        print(f"\n{RED}ERROR: Dataset not found at {data_path}{RESET}")
        sys.exit(1)

    df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} flows ({df['Label'].nunique()} classes)")

    # Clean data
    df = df.replace([float('inf'), float('-inf')], 0).fillna(0)

    # Stratified sampling: proportional to class distribution
    sampled = df.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(n=min(n_samples, len(x)), random_state=seed),
        include_groups=False
    ).reset_index(drop=True)

    # Reassign Label from the index grouping
    # Since include_groups=False drops the grouping column, re-merge
    sampled = df.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(n=min(n_samples, len(x)), random_state=seed)
    ).reset_index(drop=True)

    print(f"  Sampled {len(sampled)} flows (up to {n_samples} per class)")
    return sampled


def prepare_single_payload(row):
    """Prepare a single prediction payload (named dict)."""
    return {"features": {name: float(row[name]) for name in FEATURE_NAMES_84}}


def prepare_batch_payload(rows_df):
    """Prepare a batch prediction payload (ordered list)."""
    flows = []
    for _, row in rows_df.iterrows():
        features = [float(row[name]) for name in FEATURE_NAMES_84]
        flows.append({"features": features})
    return {"flows": flows}


# ==============================================================================
# PHASE 1: SINGLE PREDICTION LATENCY
# ==============================================================================

def phase1_single_latency(api_url, df, n_requests):
    """Measure single-prediction latency over n requests."""
    print_header("Phase 1: Single Prediction Latency")
    print(f"  Sending {n_requests} sequential requests to /predict...")

    # Use first n_requests rows (cycle if needed)
    latencies = []
    errors = 0

    for i in range(n_requests):
        row = df.iloc[i % len(df)]
        payload = prepare_single_payload(row)

        try:
            start = time.perf_counter()
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=30)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                latencies.append(elapsed_ms)
            else:
                errors += 1

        except Exception:
            errors += 1

        # Progress
        if (i + 1) % 10 == 0 or i == n_requests - 1:
            print(f"    [{i+1}/{n_requests}] completed", end="\r")

    print()

    if not latencies:
        print(f"  {RED}All requests failed!{RESET}")
        return []

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    results = {
        "avg": statistics.mean(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "p50": latencies_sorted[int(n * 0.50)],
        "p95": latencies_sorted[int(n * 0.95)],
        "p99": latencies_sorted[min(int(n * 0.99), n - 1)],
        "std": statistics.stdev(latencies) if n > 1 else 0,
        "errors": errors,
        "total": n_requests,
    }

    print(f"\n  {BOLD}Single Prediction Latency ({n} successful requests):{RESET}")
    print_metric("Average", f"{results['avg']:.1f}", "ms")
    print_metric("Std Dev", f"{results['std']:.1f}", "ms")
    print_metric("P50 (median)", f"{results['p50']:.1f}", "ms")
    print_metric("P95", f"{results['p95']:.1f}", "ms")
    print_metric("P99", f"{results['p99']:.1f}", "ms")
    print_metric("Min", f"{results['min']:.1f}", "ms")
    print_metric("Max", f"{results['max']:.1f}", "ms")
    if errors:
        print_metric("Errors", f"{errors}", f"/ {n_requests}")

    return latencies, results


# ==============================================================================
# PHASE 2: BATCH THROUGHPUT
# ==============================================================================

def phase2_batch_throughput(api_url, df):
    """Measure batch prediction throughput for various batch sizes."""
    print_header("Phase 2: Batch Throughput")

    batch_sizes = [10, 50, 100, 500]
    batch_results = []

    for batch_size in batch_sizes:
        actual_size = min(batch_size, len(df))
        sample = df.head(actual_size)
        payload = prepare_batch_payload(sample)

        print(f"\n  Batch size: {actual_size}...")

        try:
            start = time.perf_counter()
            resp = requests.post(f"{api_url}/batch_predict", json=payload, timeout=120)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                data = resp.json()
                n_predictions = len(data.get("predictions", []))
                flows_per_sec = (n_predictions / elapsed_ms) * 1000
                per_flow_ms = elapsed_ms / n_predictions if n_predictions else 0

                result = {
                    "batch_size": actual_size,
                    "total_ms": elapsed_ms,
                    "per_flow_ms": per_flow_ms,
                    "flows_per_sec": flows_per_sec,
                    "n_predictions": n_predictions,
                }
                batch_results.append(result)

                print_metric("Total time", f"{elapsed_ms:.1f}", "ms")
                print_metric("Per-flow latency", f"{per_flow_ms:.2f}", "ms")
                print_metric("Throughput", f"{flows_per_sec:.1f}", "flows/sec")
            else:
                print(f"    {RED}HTTP {resp.status_code}: {resp.text[:100]}{RESET}")

        except Exception as e:
            print(f"    {RED}Error: {e}{RESET}")

    if batch_results:
        print(f"\n  {BOLD}Batch Throughput Summary:{RESET}")
        print(f"    {'Batch Size':>12} {'Total (ms)':>12} {'Per-flow (ms)':>14} {'Flows/sec':>12}")
        print(f"    {'-'*54}")
        for r in batch_results:
            print(f"    {r['batch_size']:>12} {r['total_ms']:>12.1f} {r['per_flow_ms']:>14.2f} {r['flows_per_sec']:>12.1f}")

    return batch_results


# ==============================================================================
# PHASE 3: CONCURRENT THROUGHPUT
# ==============================================================================

def phase3_concurrent_throughput(api_url, df, n_requests):
    """Measure concurrent throughput with multiple threads."""
    print_header("Phase 3: Concurrent Throughput")

    # Check rate limit status
    try:
        resp = requests.get(f"{api_url}/security", timeout=5)
        if resp.status_code == 200:
            security = resp.json()
            rl_status = security.get("rate_limiting", {})
            if rl_status.get("enabled", False):
                print(f"\n  {YELLOW}WARNING: Rate limiting is ENABLED.{RESET}")
                print(f"  Results may be affected by rate limits ({rl_status}).")
                print("  To disable: set NIDS_RATE_LIMIT_ENABLED=false before starting API")
    except Exception:
        pass

    thread_counts = [4, 8, 16]
    concurrent_results = []

    def send_single_request(idx):
        """Worker function for a single prediction."""
        row = df.iloc[idx % len(df)]
        payload = prepare_single_payload(row)
        try:
            start = time.perf_counter()
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=30)
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, resp.status_code == 200
        except Exception:
            return 0, False

    for n_threads in thread_counts:
        print(f"\n  {n_threads} threads x {n_requests} requests...")

        latencies = []
        errors = 0

        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(send_single_request, i) for i in range(n_requests)]
            for future in as_completed(futures):
                elapsed, success = future.result()
                if success:
                    latencies.append(elapsed)
                else:
                    errors += 1

        wall_time_ms = (time.perf_counter() - wall_start) * 1000
        successful = len(latencies)
        peak_flows_sec = (successful / wall_time_ms) * 1000 if wall_time_ms > 0 else 0
        avg_latency = statistics.mean(latencies) if latencies else 0

        result = {
            "threads": n_threads,
            "total_requests": n_requests,
            "successful": successful,
            "errors": errors,
            "wall_time_ms": wall_time_ms,
            "avg_latency_ms": avg_latency,
            "peak_flows_sec": peak_flows_sec,
        }
        concurrent_results.append(result)

        print_metric("Wall time", f"{wall_time_ms:.0f}", "ms")
        print_metric("Avg latency", f"{avg_latency:.1f}", "ms")
        print_metric("Peak throughput", f"{peak_flows_sec:.1f}", "flows/sec")
        if errors:
            print_metric("Errors", f"{errors}", f"/ {n_requests}")

    if concurrent_results:
        print(f"\n  {BOLD}Concurrent Throughput Summary:{RESET}")
        print(f"    {'Threads':>8} {'Requests':>10} {'Wall (ms)':>10} {'Avg Lat (ms)':>14} {'Flows/sec':>12} {'Errors':>8}")
        print(f"    {'-'*66}")
        for r in concurrent_results:
            print(f"    {r['threads']:>8} {r['total_requests']:>10} {r['wall_time_ms']:>10.0f} "
                  f"{r['avg_latency_ms']:>14.1f} {r['peak_flows_sec']:>12.1f} {r['errors']:>8}")

    return concurrent_results


# ==============================================================================
# PHASE 4: SINGLE VS BATCH COMPARISON
# ==============================================================================

def phase4_single_vs_batch(api_url, df, n_compare=100):
    """Compare sequential single predictions vs one batch prediction."""
    print_header("Phase 4: Single vs Batch Comparison")

    actual_n = min(n_compare, len(df))
    sample = df.head(actual_n)
    print(f"  Comparing {actual_n} flows: sequential /predict vs single /batch_predict...")

    # --- Sequential single predictions ---
    print(f"\n  Sequential /predict ({actual_n} requests)...")
    single_latencies = []

    for i, (_, row) in enumerate(sample.iterrows()):
        payload = prepare_single_payload(row)
        try:
            start = time.perf_counter()
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=30)
            elapsed = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                single_latencies.append(elapsed)
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{actual_n}]", end="\r")

    single_total_ms = sum(single_latencies)
    print(f"    Total: {single_total_ms:.0f}ms ({len(single_latencies)} successful)")

    # --- Single batch prediction ---
    print(f"\n  Batch /batch_predict ({actual_n} flows in 1 request)...")
    payload = prepare_batch_payload(sample)

    try:
        start = time.perf_counter()
        resp = requests.post(f"{api_url}/batch_predict", json=payload, timeout=120)
        batch_total_ms = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            batch_n = len(data.get("predictions", []))
            print(f"    Total: {batch_total_ms:.0f}ms ({batch_n} predictions)")
        else:
            print(f"    {RED}HTTP {resp.status_code}{RESET}")
            batch_total_ms = 0
    except Exception as e:
        print(f"    {RED}Error: {e}{RESET}")
        batch_total_ms = 0

    # --- Comparison ---
    comparison = {
        "n_flows": actual_n,
        "single_total_ms": single_total_ms,
        "single_per_flow_ms": single_total_ms / len(single_latencies) if single_latencies else 0,
        "batch_total_ms": batch_total_ms,
        "batch_per_flow_ms": batch_total_ms / actual_n if batch_total_ms > 0 else 0,
        "speedup": single_total_ms / batch_total_ms if batch_total_ms > 0 else 0,
    }

    print(f"\n  {BOLD}Comparison ({actual_n} flows):{RESET}")
    print_metric("Sequential total", f"{comparison['single_total_ms']:.0f}", "ms")
    print_metric("Sequential per-flow", f"{comparison['single_per_flow_ms']:.1f}", "ms")
    print_metric("Batch total", f"{comparison['batch_total_ms']:.0f}", "ms")
    print_metric("Batch per-flow", f"{comparison['batch_per_flow_ms']:.2f}", "ms")

    if comparison['speedup'] > 0:
        color = GREEN if comparison['speedup'] > 1 else RED
        print(f"\n    {BOLD}Speedup: {color}{comparison['speedup']:.1f}x{RESET} "
              f"(batch is {comparison['speedup']:.1f}x faster)")

    return comparison


# ==============================================================================
# RESULT SAVING
# ==============================================================================

def save_results(phase1, phase2, phase3, phase4, timestamp):
    """Save benchmark results to text report and CSV."""
    results_dir = REPORTS_DIR / "benchmarks"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Text report ---
    report_path = results_dir / f"benchmark_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("NIDS Benchmark Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Phase 1
        if phase1:
            _, results = phase1
            f.write("Phase 1: Single Prediction Latency\n")
            f.write(f"{'-'*40}\n")
            f.write(f"  Requests:  {results['total']}\n")
            f.write(f"  Average:   {results['avg']:.1f} ms\n")
            f.write(f"  Std Dev:   {results['std']:.1f} ms\n")
            f.write(f"  P50:       {results['p50']:.1f} ms\n")
            f.write(f"  P95:       {results['p95']:.1f} ms\n")
            f.write(f"  P99:       {results['p99']:.1f} ms\n")
            f.write(f"  Min:       {results['min']:.1f} ms\n")
            f.write(f"  Max:       {results['max']:.1f} ms\n")
            f.write(f"  Errors:    {results['errors']}\n\n")

        # Phase 2
        if phase2:
            f.write("Phase 2: Batch Throughput\n")
            f.write(f"{'-'*40}\n")
            f.write(f"  {'Batch Size':>12} {'Total (ms)':>12} {'Per-flow (ms)':>14} {'Flows/sec':>12}\n")
            for r in phase2:
                f.write(f"  {r['batch_size']:>12} {r['total_ms']:>12.1f} "
                        f"{r['per_flow_ms']:>14.2f} {r['flows_per_sec']:>12.1f}\n")
            f.write("\n")

        # Phase 3
        if phase3:
            f.write("Phase 3: Concurrent Throughput\n")
            f.write(f"{'-'*40}\n")
            f.write(f"  {'Threads':>8} {'Wall (ms)':>10} {'Avg Lat (ms)':>14} "
                    f"{'Flows/sec':>12} {'Errors':>8}\n")
            for r in phase3:
                f.write(f"  {r['threads']:>8} {r['wall_time_ms']:>10.0f} "
                        f"{r['avg_latency_ms']:>14.1f} {r['peak_flows_sec']:>12.1f} "
                        f"{r['errors']:>8}\n")
            f.write("\n")

        # Phase 4
        if phase4:
            f.write("Phase 4: Single vs Batch Comparison\n")
            f.write(f"{'-'*40}\n")
            f.write(f"  Flows compared:      {phase4['n_flows']}\n")
            f.write(f"  Sequential total:    {phase4['single_total_ms']:.0f} ms\n")
            f.write(f"  Sequential per-flow: {phase4['single_per_flow_ms']:.1f} ms\n")
            f.write(f"  Batch total:         {phase4['batch_total_ms']:.0f} ms\n")
            f.write(f"  Batch per-flow:      {phase4['batch_per_flow_ms']:.2f} ms\n")
            f.write(f"  Speedup:             {phase4['speedup']:.1f}x\n")

    print(f"\n  {GREEN}Report saved:{RESET}  {report_path}")

    # --- CSV with raw measurements ---
    csv_path = results_dir / f"benchmark_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_phase", "metric", "value", "unit"])

        if phase1:
            _, r = phase1
            for key in ["avg", "std", "p50", "p95", "p99", "min", "max"]:
                writer.writerow(["single_latency", key, f"{r[key]:.2f}", "ms"])
            writer.writerow(["single_latency", "errors", r["errors"], "count"])
            writer.writerow(["single_latency", "total_requests", r["total"], "count"])

        if phase2:
            for r in phase2:
                writer.writerow(["batch_throughput", f"batch_{r['batch_size']}_total_ms",
                                 f"{r['total_ms']:.2f}", "ms"])
                writer.writerow(["batch_throughput", f"batch_{r['batch_size']}_per_flow_ms",
                                 f"{r['per_flow_ms']:.3f}", "ms"])
                writer.writerow(["batch_throughput", f"batch_{r['batch_size']}_flows_sec",
                                 f"{r['flows_per_sec']:.1f}", "flows/sec"])

        if phase3:
            for r in phase3:
                t = r["threads"]
                writer.writerow(["concurrent", f"{t}t_wall_ms", f"{r['wall_time_ms']:.0f}", "ms"])
                writer.writerow(["concurrent", f"{t}t_avg_latency_ms",
                                 f"{r['avg_latency_ms']:.1f}", "ms"])
                writer.writerow(["concurrent", f"{t}t_flows_sec",
                                 f"{r['peak_flows_sec']:.1f}", "flows/sec"])
                writer.writerow(["concurrent", f"{t}t_errors", r["errors"], "count"])

        if phase4:
            writer.writerow(["comparison", "sequential_total_ms",
                             f"{phase4['single_total_ms']:.0f}", "ms"])
            writer.writerow(["comparison", "batch_total_ms",
                             f"{phase4['batch_total_ms']:.0f}", "ms"])
            writer.writerow(["comparison", "speedup",
                             f"{phase4['speedup']:.1f}", "x"])

    print(f"  {GREEN}CSV saved:{RESET}     {csv_path}")
    return report_path, csv_path


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="NIDS Benchmark Suite")
    parser.add_argument("--url", default=DEFAULT_API_URL,
                        help=f"API URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--data", default=str(DATA_PATH),
                        help=f"Flows to send, Parquet or CSV (default: {DATA_PATH})")
    parser.add_argument("--samples", type=int, default=500,
                        help="Max samples per class for data pool (default: 500)")
    parser.add_argument("--single-n", type=int, default=50,
                        help="Number of single-prediction requests in Phase 1 (default: 50)")
    parser.add_argument("--concurrent-n", type=int, default=100,
                        help="Requests per thread count in Phase 3 (default: 100)")
    parser.add_argument("--skip-concurrent", action="store_true",
                        help="Skip concurrent throughput test (Phase 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data sampling (default: 42)")
    args = parser.parse_args()

    print_header("NIDS BENCHMARK SUITE")
    print(f"  API:            {args.url}")
    print(f"  Data:           {args.data}")
    print(f"  Pool size:      up to {args.samples} per class")
    print(f"  Single requests: {args.single_n}")
    print(f"  Concurrent:     {'skipped' if args.skip_concurrent else f'{args.concurrent_n} per thread count'}")

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
        print("  Start it with: python api/app.py")
        sys.exit(1)

    # Load data
    df = load_benchmark_data(Path(args.data), args.samples, args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Phase 1: Single Prediction Latency
    phase1 = phase1_single_latency(args.url, df, args.single_n)

    # Phase 2: Batch Throughput
    phase2 = phase2_batch_throughput(args.url, df)

    # Phase 3: Concurrent Throughput
    phase3 = None
    if not args.skip_concurrent:
        phase3 = phase3_concurrent_throughput(args.url, df, args.concurrent_n)
    else:
        print_header("Phase 3: Concurrent Throughput (SKIPPED)")
        print("  Use --skip-concurrent=false to enable")

    # Phase 4: Single vs Batch Comparison
    phase4 = phase4_single_vs_batch(args.url, df)

    # Save results
    print_header("Saving Results")
    save_results(phase1, phase2, phase3, phase4, timestamp)

    print(f"\n{BOLD}Benchmark complete.{RESET}\n")


if __name__ == "__main__":
    main()
