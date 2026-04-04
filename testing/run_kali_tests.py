"""
Kali VM Attack Test Orchestrator
==================================
Semi-automated script that guides you through running attacks from a Kali VM
while the NIDS engine captures and classifies traffic.

For each attack:
  1. Shows the Kali command to run
  2. Waits for you to press Enter (start the attack)
  3. Counts down the test duration
  4. Records the time window for log correlation

After all tests, saves a test plan JSON that parse_kali_results.py uses
to match predictions against expected attack types.

Usage:
    python testing/run_kali_tests.py --host-ip 192.168.56.1
    python testing/run_kali_tests.py --host-ip 192.168.56.1 --tests 1,3,8
    python testing/run_kali_tests.py --host-ip 192.168.56.1 --skip 6,7
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_API_URL = "http://localhost:8000"
RESULTS_DIR = Path(__file__).parent.parent / "results"

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_header(text):
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")


# ==============================================================================
# ATTACK DEFINITIONS
# ==============================================================================

def get_attack_tests(host_ip):
    """Return the full list of attack test definitions."""
    return [
        {
            "id": 1,
            "name": "Portscan (SYN Scan)",
            "expected_class": "Portscan",
            "kali_command": f"sudo nmap -sS -p 1-1000 {host_ip}",
            "duration_seconds": 60,
            "description": "SYN scan of first 1000 ports. Generates many 1-2 packet flows with SYN flags.",
            "category": "reconnaissance",
        },
        {
            "id": 2,
            "name": "Portscan (Version Scan)",
            "expected_class": "Portscan",
            "kali_command": f"sudo nmap -sV -p 21,22,80,8000,8080 {host_ip}",
            "duration_seconds": 60,
            "description": "Version detection on known open ports. Slightly larger flows with service probes.",
            "category": "reconnaissance",
        },
        {
            "id": 3,
            "name": "DDoS (SYN Flood)",
            "expected_class": "DDoS",
            "kali_command": f"sudo hping3 -S --flood -p 8080 {host_ip}",
            "duration_seconds": 60,
            "description": "SYN flood on port 8080. Extreme packet rate, tiny packets. Press Ctrl+C in Kali to stop.",
            "category": "dos",
        },
        {
            "id": 4,
            "name": "DoS Hulk",
            "expected_class": "DoS Hulk",
            "kali_command": f"python3 hulk.py http://{host_ip}:8080",
            "alt_command": f"pip3 install hulk && hulk http://{host_ip}:8080",
            "duration_seconds": 60,
            "description": "HTTP POST flood with random parameters. Targets the vulnerable webapp.",
            "category": "dos",
        },
        {
            "id": 5,
            "name": "DoS GoldenEye",
            "expected_class": "DoS GoldenEye",
            "kali_command": f"goldeneye http://{host_ip}:8080 -w 50 -s 75",
            "alt_command": f"git clone https://github.com/jseidl/GoldenEye && python3 GoldenEye/goldeneye.py http://{host_ip}:8080",
            "duration_seconds": 60,
            "description": "HTTP GET/POST flood with keep-alive. 50 workers, 75 sockets each.",
            "category": "dos",
        },
        {
            "id": 6,
            "name": "DoS Slowloris",
            "expected_class": "DoS Slowloris",
            "kali_command": f"slowloris {host_ip} -p 8080 -s 200",
            "alt_command": f"pip3 install slowloris && slowloris {host_ip} -p 8080 -s 200",
            "duration_seconds": 120,
            "description": "Slow HTTP headers attack. Opens 200 connections, sends partial headers. Needs longer duration.",
            "category": "dos",
        },
        {
            "id": 7,
            "name": "DoS Slowhttptest",
            "expected_class": "DoS Slowhttptest",
            "kali_command": f"slowhttptest -c 500 -H -i 10 -r 200 -t GET -u http://{host_ip}:8080 -p 3",
            "duration_seconds": 120,
            "description": "Slow HTTP test tool. 500 connections, 10s interval. Install: apt install slowhttptest",
            "category": "dos",
        },
        {
            "id": 8,
            "name": "FTP Brute Force",
            "expected_class": "FTP-Patator",
            "kali_command": f"hydra -l admin -P /usr/share/wordlists/rockyou.txt -t 16 ftp://{host_ip}",
            "duration_seconds": 60,
            "description": "FTP brute force with hydra. 16 threads. Target must have FTP server running (port 21).",
            "category": "brute_force",
            "requires": "FTP server on port 21",
        },
        {
            "id": 9,
            "name": "SSH Brute Force",
            "expected_class": "SSH-Patator",
            "kali_command": f"hydra -l admin -P /usr/share/wordlists/rockyou.txt -t 8 ssh://{host_ip}",
            "duration_seconds": 60,
            "description": "SSH brute force with hydra. 8 threads. Target must have SSH server running (port 22).",
            "category": "brute_force",
            "requires": "SSH server on port 22",
        },
        {
            "id": 10,
            "name": "Web Attack - Brute Force",
            "expected_class": "Web Attack - Brute Force",
            "kali_command": (
                f'hydra -l admin -P /usr/share/wordlists/rockyou.txt '
                f'{host_ip} http-post-form "/login:username=^USER^&password=^PASS^:Invalid" -t 16 -s 8080'
            ),
            "duration_seconds": 60,
            "description": "HTTP form brute force against /login endpoint. 16 threads.",
            "category": "web_attack",
        },
        {
            "id": 11,
            "name": "Web Attack - SQL Injection",
            "expected_class": "Web Attack - SQL Injection",
            "kali_command": f'sqlmap -u "http://{host_ip}:8080/users?id=1" --batch --level=3',
            "duration_seconds": 90,
            "description": "Automated SQL injection testing against /users?id= endpoint.",
            "category": "web_attack",
        },
        {
            "id": 12,
            "name": "Web Attack - XSS",
            "expected_class": "Web Attack - XSS",
            "kali_command": f'xsser -u "http://{host_ip}:8080/search?q=test" --auto',
            "alt_command": f'python3 -c "import requests; [requests.get(\'http://{host_ip}:8080/search\', params={{\'q\': f\'<script>alert({{i}})</script>\'}}) for i in range(200)]"',
            "duration_seconds": 90,
            "description": "XSS payload testing against /search?q= endpoint. Alt: manual script injection loop.",
            "category": "web_attack",
        },
    ]


# ==============================================================================
# TEST RUNNER
# ==============================================================================

def check_nids_running(api_url):
    """Verify the NIDS API is reachable."""
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        health = resp.json()
        if health.get("models_loaded"):
            print(f"  {GREEN}API online{RESET} (uptime: {health.get('uptime_seconds', 0):.0f}s)")
            return True
        else:
            print(f"  {RED}API online but models not loaded!{RESET}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  {RED}Cannot reach API at {api_url}{RESET}")
        print(f"  Start NIDS first: python run_nids.py -i \"<interface>\"")
        return False


def countdown(seconds, label=""):
    """Visual countdown timer."""
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        timer = f"{mins:02d}:{secs:02d}"
        print(f"\r  {DIM}[{timer}] {label}{RESET}    ", end="", flush=True)
        time.sleep(1)
    print(f"\r  {DIM}[00:00] Done{RESET}                    ")


def run_test(test, test_results):
    """Run a single attack test."""
    print(f"\n  {BOLD}Test #{test['id']}: {test['name']}{RESET}")
    print(f"  Expected class: {YELLOW}{test['expected_class']}{RESET}")
    print(f"  Category:       {test['category']}")
    print(f"  Duration:       {test['duration_seconds']}s")

    if test.get("requires"):
        print(f"  Requires:       {test['requires']}")

    print(f"\n  {BOLD}Run this command in Kali:{RESET}")
    print(f"  {CYAN}{test['kali_command']}{RESET}")

    if test.get("alt_command"):
        print(f"  {DIM}Alternative: {test['alt_command']}{RESET}")

    print(f"\n  {test['description']}")

    # Wait for user to start the attack
    print(f"\n  {YELLOW}Press Enter to start, or type 's' to skip this test:{RESET} ", end="")
    try:
        response = input().strip().lower()
    except EOFError:
        return None
    if response == 's':
        print(f"  {DIM}Skipped.{RESET}")
        return None

    # Record start time
    start_time = datetime.now()
    start_iso = start_time.isoformat()
    print(f"  {GREEN}Started at {start_time.strftime('%H:%M:%S')}{RESET}")

    # Countdown
    countdown(test["duration_seconds"], f"Capturing {test['name']} traffic...")

    # Record end time
    end_time = datetime.now()
    end_iso = end_time.isoformat()

    print(f"  {GREEN}Window: {start_time.strftime('%H:%M:%S')} -> {end_time.strftime('%H:%M:%S')}{RESET}")
    print(f"\n  {YELLOW}Stop the attack in Kali now (Ctrl+C).{RESET}")
    print(f"  {YELLOW}Press Enter when ready to continue...{RESET}", end="")
    try:
        input()
    except EOFError:
        pass

    result = {
        "test_id": test["id"],
        "name": test["name"],
        "expected_class": test["expected_class"],
        "category": test["category"],
        "kali_command": test["kali_command"],
        "start_time": start_iso,
        "end_time": end_iso,
        "duration_seconds": test["duration_seconds"],
    }

    test_results.append(result)
    return result


def run_all_tests(tests, api_url):
    """Run all selected tests sequentially."""
    test_results = []
    session_start = datetime.now()

    print_header("Kali VM Attack Test Session")
    print(f"  Session started: {session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Tests to run:    {len(tests)}")
    print(f"  API:             {api_url}")

    # Pre-check
    print(f"\n  Checking NIDS status...")
    if not check_nids_running(api_url):
        sys.exit(1)

    # Run each test
    for i, test in enumerate(tests):
        print_header(f"Test {i+1}/{len(tests)}: {test['name']}")
        run_test(test, test_results)

    # Wait for flow timeout
    print_header("Waiting for Flow Processing")
    print(f"  Flows need up to 120s to timeout and be sent for prediction.")
    print(f"  Waiting 130s to ensure all flows are processed...")
    countdown(130, "Waiting for remaining flows to complete...")

    session_end = datetime.now()

    # Save test plan
    test_plan = {
        "session_start": session_start.isoformat(),
        "session_end": session_end.isoformat(),
        "api_url": api_url,
        "total_tests": len(test_results),
        "tests": test_results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = session_start.strftime("%Y%m%d_%H%M%S")
    plan_path = RESULTS_DIR / f"kali_test_plan_{timestamp}.json"

    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(test_plan, f, indent=2)

    print_header("Test Session Complete")
    print(f"  Duration:    {(session_end - session_start).total_seconds():.0f}s")
    print(f"  Tests run:   {len(test_results)}")
    print(f"  Test plan:   {GREEN}{plan_path}{RESET}")
    print(f"\n  {BOLD}Next step:{RESET} Parse the results:")
    print(f"  python testing/parse_kali_results.py --plan \"{plan_path}\"")

    return test_plan, plan_path


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kali VM attack test orchestrator for NIDS evaluation"
    )
    parser.add_argument("--host-ip", required=True,
                        help="IP address of this machine (visible to Kali VM)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                        help=f"NIDS API URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--tests", default=None,
                        help="Comma-separated test IDs to run (e.g., '1,3,8')")
    parser.add_argument("--skip", default=None,
                        help="Comma-separated test IDs to skip (e.g., '6,7')")
    parser.add_argument("--list", action="store_true",
                        help="List all available tests and exit")
    args = parser.parse_args()

    all_tests = get_attack_tests(args.host_ip)

    # List mode
    if args.list:
        print_header("Available Attack Tests")
        for t in all_tests:
            req = f"  (Requires: {t['requires']})" if t.get("requires") else ""
            print(f"  [{t['id']:>2}] {t['name']:<35} -> {t['expected_class']}{req}")
        return

    # Filter tests
    tests = all_tests

    if args.tests:
        test_ids = [int(x.strip()) for x in args.tests.split(",")]
        tests = [t for t in tests if t["id"] in test_ids]
        if not tests:
            print(f"{RED}No matching tests found for IDs: {args.tests}{RESET}")
            sys.exit(1)

    if args.skip:
        skip_ids = [int(x.strip()) for x in args.skip.split(",")]
        tests = [t for t in tests if t["id"] not in skip_ids]

    print_header("NIDS KALI VM ATTACK TESTING")
    print(f"\n  Host IP:     {args.host_ip}")
    print(f"  API URL:     {args.api_url}")
    print(f"  Tests:       {len(tests)}")
    print(f"\n  {BOLD}Tests to run:{RESET}")
    for t in tests:
        print(f"    [{t['id']:>2}] {t['name']:<35} ({t['duration_seconds']}s)")

    print(f"\n  {BOLD}Checklist before starting:{RESET}")
    print(f"    [?] NIDS engine running: python run_nids.py -i \"<host-only adapter>\"")
    print(f"    [?] Target services running: python testing/setup_services.py")
    print(f"    [?] Kali VM can reach {args.host_ip}: ping {args.host_ip}")
    print(f"    [?] Rate limiting disabled: set NIDS_RATE_LIMIT_ENABLED=false")

    print(f"\n  {YELLOW}Press Enter to begin testing...{RESET}", end="")
    try:
        input()
    except EOFError:
        return

    run_all_tests(tests, args.api_url)


if __name__ == "__main__":
    main()
