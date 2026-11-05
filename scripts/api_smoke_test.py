"""
NIDS API Test Client
====================
Test script to verify all API endpoints are working correctly.

Usage:
    python test_client.py
    python test_client.py --url http://localhost:8000
"""

import argparse
import sys
import time
from typing import Any

try:
    import requests
except ImportError:
    print("Error: requests library not installed. Run: pip install requests")
    sys.exit(1)


# Default API URL
DEFAULT_URL = "http://localhost:8000"

# Sample network flow with 84 features (typical BENIGN traffic)
SAMPLE_BENIGN_FLOW = {
    "Src Port": 49234,
    "Dst Port": 443,
    "Protocol": 6,
    "Flow Duration": 1234567,
    "Total Fwd Packet": 10,
    "Total Bwd packets": 8,
    "Total Length of Fwd Packet": 1500,
    "Total Length of Bwd Packet": 3200,
    "Fwd Packet Length Max": 1460,
    "Fwd Packet Length Min": 0,
    "Fwd Packet Length Mean": 150.0,
    "Fwd Packet Length Std": 200.5,
    "Bwd Packet Length Max": 1460,
    "Bwd Packet Length Min": 0,
    "Bwd Packet Length Mean": 400.0,
    "Bwd Packet Length Std": 350.2,
    "Flow Bytes/s": 5000.0,
    "Flow Packets/s": 50.0,
    "Flow IAT Mean": 50000.0,
    "Flow IAT Std": 25000.0,
    "Flow IAT Max": 200000,
    "Flow IAT Min": 1000,
    "Fwd IAT Total": 500000,
    "Fwd IAT Mean": 50000.0,
    "Fwd IAT Std": 20000.0,
    "Fwd IAT Max": 150000,
    "Fwd IAT Min": 500,
    "Bwd IAT Total": 400000,
    "Bwd IAT Mean": 50000.0,
    "Bwd IAT Std": 18000.0,
    "Bwd IAT Max": 120000,
    "Bwd IAT Min": 800,
    "Fwd PSH Flags": 1,
    "Bwd PSH Flags": 1,
    "Fwd URG Flags": 0,
    "Bwd URG Flags": 0,
    "Fwd RST Flags": 0,
    "Bwd RST Flags": 0,
    "Fwd Header Length": 320,
    "Bwd Header Length": 256,
    "Fwd Packets/s": 25.0,
    "Bwd Packets/s": 20.0,
    "Packet Length Min": 0,
    "Packet Length Max": 1460,
    "Packet Length Mean": 260.0,
    "Packet Length Std": 280.5,
    "Packet Length Variance": 78680.25,
    "FIN Flag Count": 1,
    "SYN Flag Count": 1,
    "RST Flag Count": 0,
    "PSH Flag Count": 2,
    "ACK Flag Count": 16,
    "URG Flag Count": 0,
    "CWR Flag Count": 0,
    "ECE Flag Count": 0,
    "Down/Up Ratio": 0.8,
    "Average Packet Size": 261.0,
    "Fwd Segment Size Avg": 150.0,
    "Bwd Segment Size Avg": 400.0,
    "Fwd Bytes/Bulk Avg": 0,
    "Fwd Packet/Bulk Avg": 0,
    "Fwd Bulk Rate Avg": 0,
    "Bwd Bytes/Bulk Avg": 0,
    "Bwd Packet/Bulk Avg": 0,
    "Bwd Bulk Rate Avg": 0,
    "Subflow Fwd Packets": 10,
    "Subflow Fwd Bytes": 1500,
    "Subflow Bwd Packets": 8,
    "Subflow Bwd Bytes": 3200,
    "FWD Init Win Bytes": 65535,
    "Bwd Init Win Bytes": 65535,
    "Fwd Act Data Pkts": 5,
    "Fwd Seg Size Min": 20,
    "Active Mean": 100000.0,
    "Active Std": 50000.0,
    "Active Max": 200000,
    "Active Min": 10000,
    "Idle Mean": 500000.0,
    "Idle Std": 250000.0,
    "Idle Max": 1000000,
    "Idle Min": 100000,
    "ICMP Code": 0,
    "ICMP Type": 0,
    "Total TCP Flow Time": 1234567
}

# Sample attack flow (DDoS-like characteristics)
SAMPLE_ATTACK_FLOW = {
    "Src Port": 12345,
    "Dst Port": 80,
    "Protocol": 6,
    "Flow Duration": 50000,
    "Total Fwd Packet": 1000,
    "Total Bwd packets": 5,
    "Total Length of Fwd Packet": 60000,
    "Total Length of Bwd Packet": 200,
    "Fwd Packet Length Max": 60,
    "Fwd Packet Length Min": 60,
    "Fwd Packet Length Mean": 60.0,
    "Fwd Packet Length Std": 0.0,
    "Bwd Packet Length Max": 40,
    "Bwd Packet Length Min": 40,
    "Bwd Packet Length Mean": 40.0,
    "Bwd Packet Length Std": 0.0,
    "Flow Bytes/s": 1200000.0,
    "Flow Packets/s": 20000.0,
    "Flow IAT Mean": 50.0,
    "Flow IAT Std": 10.0,
    "Flow IAT Max": 100,
    "Flow IAT Min": 10,
    "Fwd IAT Total": 50000,
    "Fwd IAT Mean": 50.0,
    "Fwd IAT Std": 10.0,
    "Fwd IAT Max": 100,
    "Fwd IAT Min": 10,
    "Bwd IAT Total": 1000,
    "Bwd IAT Mean": 200.0,
    "Bwd IAT Std": 50.0,
    "Bwd IAT Max": 500,
    "Bwd IAT Min": 100,
    "Fwd PSH Flags": 0,
    "Bwd PSH Flags": 0,
    "Fwd URG Flags": 0,
    "Bwd URG Flags": 0,
    "Fwd RST Flags": 0,
    "Bwd RST Flags": 0,
    "Fwd Header Length": 20000,
    "Bwd Header Length": 100,
    "Fwd Packets/s": 20000.0,
    "Bwd Packets/s": 100.0,
    "Packet Length Min": 40,
    "Packet Length Max": 60,
    "Packet Length Mean": 59.0,
    "Packet Length Std": 2.0,
    "Packet Length Variance": 4.0,
    "FIN Flag Count": 0,
    "SYN Flag Count": 1000,
    "RST Flag Count": 0,
    "PSH Flag Count": 0,
    "ACK Flag Count": 0,
    "URG Flag Count": 0,
    "CWR Flag Count": 0,
    "ECE Flag Count": 0,
    "Down/Up Ratio": 0.005,
    "Average Packet Size": 59.0,
    "Fwd Segment Size Avg": 60.0,
    "Bwd Segment Size Avg": 40.0,
    "Fwd Bytes/Bulk Avg": 0,
    "Fwd Packet/Bulk Avg": 0,
    "Fwd Bulk Rate Avg": 0,
    "Bwd Bytes/Bulk Avg": 0,
    "Bwd Packet/Bulk Avg": 0,
    "Bwd Bulk Rate Avg": 0,
    "Subflow Fwd Packets": 1000,
    "Subflow Fwd Bytes": 60000,
    "Subflow Bwd Packets": 5,
    "Subflow Bwd Bytes": 200,
    "FWD Init Win Bytes": 1024,
    "Bwd Init Win Bytes": 0,
    "Fwd Act Data Pkts": 1000,
    "Fwd Seg Size Min": 8,
    "Active Mean": 50000.0,
    "Active Std": 0.0,
    "Active Max": 50000,
    "Active Min": 50000,
    "Idle Mean": 0.0,
    "Idle Std": 0.0,
    "Idle Max": 0,
    "Idle Min": 0,
    "ICMP Code": 0,
    "ICMP Type": 0,
    "Total TCP Flow Time": 50000
}


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}[PASS]{Colors.RESET} {text}")


def print_failure(text: str):
    """Print failure message."""
    print(f"{Colors.RED}[FAIL]{Colors.RESET} {text}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.YELLOW}[INFO]{Colors.RESET} {text}")


def test_endpoint(
    method: str,
    url: str,
    name: str,
    expected_status: int = 200,
    json_data: dict = None
) -> bool:
    """
    Test an API endpoint and return success/failure.

    Args:
        method: HTTP method (GET, POST)
        url: Full endpoint URL
        name: Test name for display
        expected_status: Expected HTTP status code
        json_data: JSON payload for POST requests

    Returns:
        bool: True if test passed, False otherwise
    """
    try:
        start_time = time.time()

        if method.upper() == "GET":
            response = requests.get(url, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, json=json_data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")

        elapsed = (time.time() - start_time) * 1000

        if response.status_code == expected_status:
            print_success(f"{name} ({elapsed:.1f}ms)")
            return True, response.json() if response.text else {}
        else:
            print_failure(f"{name} - Expected {expected_status}, got {response.status_code}")
            print(f"         Response: {response.text[:200]}")
            return False, {}

    except requests.exceptions.ConnectionError:
        print_failure(f"{name} - Connection refused. Is the API running?")
        return False, {}
    except requests.exceptions.Timeout:
        print_failure(f"{name} - Request timed out")
        return False, {}
    except Exception as e:
        print_failure(f"{name} - Error: {str(e)}")
        return False, {}


def run_tests(base_url: str) -> dict[str, Any]:
    """
    Run all API tests and return results summary.

    Args:
        base_url: Base URL of the API

    Returns:
        Dict with test results summary
    """
    results = {"passed": 0, "failed": 0, "tests": []}

    print_header("NIDS API Test Suite")
    print_info(f"Testing API at: {base_url}")

    # Test 1: Root endpoint
    print_header("Test 1: Root Endpoint (GET /)")
    success, data = test_endpoint("GET", f"{base_url}/", "Root endpoint")
    if success:
        results["passed"] += 1
        print_info(f"API Version: {data.get('version', 'N/A')}")
        print_info(f"Attack classes: {len(data.get('attack_classes', []))}")
    else:
        results["failed"] += 1

    # Test 2: Health check
    print_header("Test 2: Health Check (GET /health)")
    success, data = test_endpoint("GET", f"{base_url}/health", "Health check")
    if success:
        results["passed"] += 1
        print_info(f"Status: {data.get('status', 'N/A')}")
        print_info(f"Models loaded: {data.get('models_loaded', 'N/A')}")
        print_info(f"Uptime: {data.get('uptime_seconds', 0):.1f}s")
    else:
        results["failed"] += 1

    # Test 3: Metrics
    print_header("Test 3: Metrics (GET /stats)")
    success, data = test_endpoint("GET", f"{base_url}/stats", "Metrics endpoint")
    if success:
        results["passed"] += 1
        print_info(f"Total predictions: {data.get('total_predictions', 0)}")
        print_info(f"Total attacks: {data.get('total_attacks', 0)}")
    else:
        results["failed"] += 1

    # Test 4: Feature info
    print_header("Test 4: Features Info (GET /features)")
    success, data = test_endpoint("GET", f"{base_url}/features", "Features endpoint")
    if success:
        results["passed"] += 1
        print_info(f"Total features: {data.get('total_features', 0)}")
        print_info(f"Selected features: {len(data.get('selected_feature_names', []))}")
    else:
        results["failed"] += 1

    # Test 5: Attack classes
    print_header("Test 5: Attack Classes (GET /classes)")
    success, data = test_endpoint("GET", f"{base_url}/classes", "Classes endpoint")
    if success:
        results["passed"] += 1
        print_info(f"Total classes: {data.get('total_classes', 0)}")
        classes = data.get('classes', [])
        attacks = [c['name'] for c in classes if c.get('is_attack')]
        print_info(f"Attack types: {', '.join(attacks[:5])}...")
    else:
        results["failed"] += 1

    # Test 6: Single prediction (BENIGN)
    print_header("Test 6: Single Prediction - BENIGN (POST /predict)")
    payload = {"features": SAMPLE_BENIGN_FLOW}
    success, data = test_endpoint("POST", f"{base_url}/predict", "Predict BENIGN", json_data=payload)
    if success:
        results["passed"] += 1
        print_info(f"Prediction: {data.get('prediction', 'N/A')}")
        print_info(f"Confidence: {data.get('confidence', 0):.4f}")
        print_info(f"Is attack: {data.get('is_attack', 'N/A')}")

        top_features = data.get('top_features', [])
        if top_features:
            print_info("Top contributing features:")
            for f in top_features[:3]:
                print(f"         - {f['feature']}: {f['contribution']:.4f}")
    else:
        results["failed"] += 1

    # Test 7: Single prediction (Attack)
    print_header("Test 7: Single Prediction - Attack (POST /predict)")
    payload = {"features": SAMPLE_ATTACK_FLOW}
    success, data = test_endpoint("POST", f"{base_url}/predict", "Predict Attack", json_data=payload)
    if success:
        results["passed"] += 1
        prediction = data.get('prediction', 'N/A')
        is_attack = data.get('is_attack', False)
        print_info(f"Prediction: {prediction}")
        print_info(f"Confidence: {data.get('confidence', 0):.4f}")
        print_info(f"Is attack: {is_attack}")

        if is_attack:
            print_success("Correctly identified as attack!")
        else:
            print_info("Classified as BENIGN (model may have different thresholds)")
    else:
        results["failed"] += 1

    # Test 8: Batch prediction
    print_header("Test 8: Batch Prediction (POST /batch_predict)")
    # Create batch with 3 flows
    batch_features = [list(SAMPLE_BENIGN_FLOW.values()),
                     list(SAMPLE_ATTACK_FLOW.values()),
                     list(SAMPLE_BENIGN_FLOW.values())]
    batch_payload = {
        "flows": [{"features": f} for f in batch_features]
    }
    success, data = test_endpoint("POST", f"{base_url}/batch_predict", "Batch predict", json_data=batch_payload)
    if success:
        results["passed"] += 1
        print_info(f"Total flows: {data.get('total_flows', 0)}")
        print_info(f"Attack count: {data.get('attack_count', 0)}")
        print_info(f"Processing time: {data.get('processing_time_ms', 0):.2f}ms")
    else:
        results["failed"] += 1

    # Test 9: Invalid request
    print_header("Test 9: Error Handling - Invalid Request (POST /predict)")
    invalid_payload = {"features": {"invalid": 1}}  # Only 1 feature
    success, _ = test_endpoint("POST", f"{base_url}/predict", "Invalid request (should fail)",
                               expected_status=422, json_data=invalid_payload)
    if success:
        results["passed"] += 1
        print_info("Correctly returned 422 for invalid input")
    else:
        results["failed"] += 1

    # Print summary
    print_header("Test Summary")
    total = results["passed"] + results["failed"]
    print(f"\n{Colors.BOLD}Results: {results['passed']}/{total} tests passed{Colors.RESET}")

    if results["failed"] == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! API is working correctly.{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}{results['failed']} test(s) failed.{Colors.RESET}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NIDS API Test Client")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"API base URL (default: {DEFAULT_URL})"
    )
    args = parser.parse_args()

    # Run tests
    results = run_tests(args.url)

    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
