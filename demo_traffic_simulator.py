# ============================================================================
# NIDS DEMO TRAFFIC SIMULATOR
# Interactive replay of CIC-IDS2017 flows for FYP demonstration
# Author: C00286234
# ============================================================================
#
# Sends real network flow data to the NIDS API with keyboard controls
# to trigger attack scenarios on demand.
#
# Usage:
#   1. Start API:       python api/app.py
#   2. Start Dashboard: python dashboard/app.py
#   3. Run Simulator:   python demo_traffic_simulator.py
#
# Controls:
#   [SPACE] Pause / Resume
#   [1] DDoS attack        [2] Port Scan       [3] DoS Hulk
#   [4] Brute Force         [5] Web Attacks     [6] Botnet
#   [7] All attacks mixed   [8] DoS Slowloris   [9] Infiltration
#   [0] Normal traffic only
#   [+] Speed up            [-] Slow down
#   [q] Quit
#
# ============================================================================

import os
import sys
import time
import random
import threading
import requests
import pandas as pd
from collections import deque
from datetime import datetime

# Windows keyboard input
if sys.platform == "win32":
    import msvcrt
else:
    import tty
    import termios
    import select

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data", "processed", "cicids2017_cleaned.csv")
API_URL = os.getenv("NIDS_API_URL", "http://localhost:8000")

from src.constants import FEATURE_NAMES_84

# Attack scenario key mappings
ATTACK_SCENARIOS = {
    '1': {
        'name': 'DDoS',
        'types': ['DDoS'],
        'color': '\033[91m',  # Red
    },
    '2': {
        'name': 'Port Scan',
        'types': ['Portscan', 'Infiltration - Portscan'],
        'color': '\033[93m',  # Yellow
    },
    '3': {
        'name': 'DoS Hulk',
        'types': ['DoS Hulk'],
        'color': '\033[91m',
    },
    '4': {
        'name': 'Brute Force',
        'types': ['FTP-Patator', 'SSH-Patator'],
        'color': '\033[95m',  # Magenta
    },
    '5': {
        'name': 'Web Attacks',
        'types': ['Web Attack - Brute Force', 'Web Attack - XSS', 'Web Attack - SQL Injection'],
        'color': '\033[96m',  # Cyan
    },
    '6': {
        'name': 'Botnet',
        'types': ['Botnet'],
        'color': '\033[92m',  # Green
    },
    '7': {
        'name': 'ALL ATTACKS',
        'types': '__all__',
        'color': '\033[91m',
    },
    '8': {
        'name': 'DoS Slowloris',
        'types': ['DoS Slowloris', 'DoS Slowhttptest', 'DoS GoldenEye'],
        'color': '\033[91m',
    },
    '9': {
        'name': 'Infiltration',
        'types': ['Infiltration', 'Infiltration - Portscan'],
        'color': '\033[93m',
    },
}

# Terminal colors
RESET = '\033[0m'
BOLD = '\033[1m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
DIM = '\033[2m'


# ============================================================================
# DATA LOADER
# ============================================================================

class DataPool:
    """Manages pools of flows by attack type for efficient sampling."""

    def __init__(self, csv_path, max_per_class=5000):
        print(f"\n  Loading dataset from {os.path.relpath(csv_path, BASE_PATH)}...")

        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c != 'Label']

        self.pools = {}
        self.all_attack_types = []

        for label, group in df.groupby('Label'):
            # Sample up to max_per_class rows per class for memory efficiency
            if len(group) > max_per_class:
                group = group.sample(n=max_per_class, random_state=42)

            rows = group[feature_cols].values.tolist()
            random.shuffle(rows)
            self.pools[label] = rows

            if label != 'BENIGN':
                self.all_attack_types.append(label)

        # Track current index per pool for round-robin sampling
        self.indices = {label: 0 for label in self.pools}

        print(f"  Loaded {len(self.pools)} traffic classes:")
        for label in sorted(self.pools.keys()):
            count = len(self.pools[label])
            tag = "NORMAL" if label == "BENIGN" else "ATTACK"
            print(f"    [{tag:6s}] {label:35s}: {count:,} samples")

    def get_flow(self, label):
        """Get next flow for a given label (round-robin)."""
        if label not in self.pools or not self.pools[label]:
            return None
        idx = self.indices[label]
        flow = self.pools[label][idx]
        self.indices[label] = (idx + 1) % len(self.pools[label])
        return flow, label

    def get_benign(self):
        """Get a normal traffic flow."""
        return self.get_flow('BENIGN')

    def get_attack(self, attack_types):
        """Get a random attack flow from the specified types."""
        if attack_types == '__all__':
            attack_types = self.all_attack_types
        available = [t for t in attack_types if t in self.pools and self.pools[t]]
        if not available:
            return self.get_benign()
        chosen = random.choice(available)
        return self.get_flow(chosen)


# ============================================================================
# SIMULATOR STATE
# ============================================================================

class SimulatorState:
    """Thread-safe shared state for the simulator."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.paused = False
        self.speed = 2.0  # flows per second
        self.mode = 'normal'  # 'normal' or attack scenario key
        self.attack_mix = 0.30  # proportion of attack traffic in attack mode

        # Stats
        self.total_sent = 0
        self.total_attacks_sent = 0
        self.total_correct = 0
        self.total_attacks_detected = 0
        self.recent = deque(maxlen=5)  # Recent prediction results
        self.errors = 0

        # Timing
        self.start_time = time.time()


# ============================================================================
# API CLIENT
# ============================================================================

def send_flow(features, api_url):
    """Send a single flow to the API and return the prediction."""
    payload = {
        "features": {name: float(val) for name, val in zip(FEATURE_NAMES_84, features)}
    }
    try:
        resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('prediction', 'UNKNOWN'), data.get('confidence', 0.0)
        else:
            return None, 0.0
    except requests.exceptions.RequestException:
        return None, 0.0


# ============================================================================
# TRAFFIC WORKER THREAD
# ============================================================================

def traffic_worker(state, data_pool, api_url):
    """Background thread that sends flows at the configured rate."""

    while state.running:
        if state.paused:
            time.sleep(0.1)
            continue

        with state.lock:
            speed = state.speed
            mode = state.mode
            attack_mix = state.attack_mix

        # Decide what to send
        if mode == 'normal':
            flow_data = data_pool.get_benign()
            true_label = 'BENIGN'
        else:
            scenario = ATTACK_SCENARIOS.get(mode)
            if scenario and random.random() < attack_mix:
                result = data_pool.get_attack(scenario['types'])
                if result:
                    flow_data = result
                    true_label = result[1]
                else:
                    flow_data = data_pool.get_benign()
                    true_label = 'BENIGN'
            else:
                flow_data = data_pool.get_benign()
                true_label = 'BENIGN'

        if flow_data is None:
            time.sleep(0.1)
            continue

        features = flow_data[0] if isinstance(flow_data, tuple) else flow_data
        if isinstance(flow_data, tuple):
            true_label = flow_data[1]

        # Send to API
        prediction, confidence = send_flow(features, api_url)

        with state.lock:
            state.total_sent += 1
            is_attack = true_label != 'BENIGN'

            if is_attack:
                state.total_attacks_sent += 1

            if prediction is None:
                state.errors += 1
                state.recent.append(('ERROR', true_label, '', 0.0, False))
            else:
                correct = (prediction == true_label) or (
                    # Accept "Rare Attack" as correct for rare classes
                    is_attack and prediction != 'BENIGN'
                )
                if correct:
                    state.total_correct += 1
                if is_attack and prediction != 'BENIGN':
                    state.total_attacks_detected += 1

                state.recent.append((
                    'OK',
                    true_label,
                    prediction,
                    confidence,
                    correct
                ))

        # Wait according to speed
        if speed > 0:
            time.sleep(1.0 / speed)


# ============================================================================
# TERMINAL DISPLAY
# ============================================================================

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if sys.platform == 'win32' else 'clear')


def render_display(state):
    """Render the live terminal UI."""

    with state.lock:
        total = state.total_sent
        attacks_sent = state.total_attacks_sent
        attacks_detected = state.total_attacks_detected
        correct = state.total_correct
        speed = state.speed
        mode = state.mode
        paused = state.paused
        errors = state.errors
        recent = list(state.recent)
        elapsed = time.time() - state.start_time

    # Calculate rates
    accuracy = (correct / total * 100) if total > 0 else 0.0
    detect_rate = (attacks_detected / attacks_sent * 100) if attacks_sent > 0 else 0.0
    elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))

    # Mode display
    if paused:
        mode_str = f"{YELLOW}PAUSED{RESET}"
    elif mode == 'normal':
        mode_str = f"{GREEN}NORMAL TRAFFIC{RESET}"
    else:
        scenario = ATTACK_SCENARIOS.get(mode, {})
        color = scenario.get('color', RED)
        name = scenario.get('name', 'Unknown')
        mode_str = f"{color}{BOLD}⚠  {name.upper()} ATTACK  ⚠{RESET}"

    # Build display
    lines = []
    lines.append("")
    lines.append(f"  {BOLD}{'═' * 60}{RESET}")
    lines.append(f"  {BOLD}  NIDS DEMO SIMULATOR{RESET}  │  Speed: {speed:.1f} flows/sec  │  {elapsed_str}")
    lines.append(f"  {BOLD}{'═' * 60}{RESET}")
    lines.append(f"")
    lines.append(f"    Mode:     {mode_str}")
    lines.append(f"    Sent:     {total:,} total  │  {attacks_sent:,} attacks  │  {errors} errors")
    lines.append(f"    Accuracy: {GREEN}{accuracy:.1f}%{RESET} overall  │  {detect_rate:.1f}% attack detection")
    lines.append(f"")
    lines.append(f"  {DIM}{'─' * 60}{RESET}")
    lines.append(f"  {BOLD}  Recent Predictions:{RESET}")

    for status, true_label, pred, conf, ok in recent:
        if status == 'ERROR':
            lines.append(f"    {RED}✗ ERROR  {true_label}{RESET}")
        elif ok:
            tag = f"{GREEN}✓{RESET}"
            lines.append(f"    {tag} {true_label:30s} → {pred} ({conf:.0%})")
        else:
            tag = f"{RED}✗{RESET}"
            lines.append(f"    {tag} {true_label:30s} → {RED}{pred}{RESET} ({conf:.0%})")

    if not recent:
        lines.append(f"    {DIM}Waiting for first prediction...{RESET}")

    lines.append(f"")
    lines.append(f"  {DIM}{'─' * 60}{RESET}")
    lines.append(f"  {BOLD}  Controls:{RESET}")
    lines.append(f"    {CYAN}[SPACE]{RESET} Pause/Resume    {CYAN}[0]{RESET} Normal    {CYAN}[q]{RESET} Quit")
    lines.append(f"    {CYAN}[1]{RESET} DDoS  {CYAN}[2]{RESET} Portscan  {CYAN}[3]{RESET} DoS Hulk  {CYAN}[4]{RESET} Brute Force")
    lines.append(f"    {CYAN}[5]{RESET} Web Attacks  {CYAN}[6]{RESET} Botnet  {CYAN}[7]{RESET} All Attacks  {CYAN}[8]{RESET} Slowloris")
    lines.append(f"    {CYAN}[+]{RESET} Faster       {CYAN}[-]{RESET} Slower  {CYAN}[9]{RESET} Infiltration")
    lines.append(f"  {BOLD}{'═' * 60}{RESET}")
    lines.append(f"")

    # Move cursor to top and redraw (avoid flicker)
    output = '\033[H' + '\033[J' + '\n'.join(lines)
    sys.stdout.write(output)
    sys.stdout.flush()


# ============================================================================
# KEYBOARD INPUT
# ============================================================================

def get_key_windows():
    """Non-blocking key read on Windows."""
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch == b'\xe0' or ch == b'\x00':
            msvcrt.getch()  # Skip extended key
            return None
        return ch.decode('utf-8', errors='ignore')
    return None


def get_key_unix():
    """Non-blocking key read on Unix/Mac."""
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1)
    return None


def get_key():
    """Cross-platform non-blocking key read."""
    if sys.platform == 'win32':
        return get_key_windows()
    else:
        return get_key_unix()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  NIDS DEMO TRAFFIC SIMULATOR{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    # Check API health
    print(f"\n  Checking API at {API_URL}...")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            health = resp.json()
            print(f"  {GREEN}[OK]{RESET} API is {health.get('status', 'running')}")
        else:
            print(f"  {YELLOW}[WARN]{RESET} API returned status {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"  {RED}[ERROR]{RESET} Cannot connect to API at {API_URL}")
        print(f"  Start the API first: python api/app.py")
        sys.exit(1)

    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"  {RED}[ERROR]{RESET} Dataset not found: {DATA_PATH}")
        print(f"  Run 02_data_preprocessing.py first.")
        sys.exit(1)

    data_pool = DataPool(DATA_PATH)

    # Initialize state
    state = SimulatorState()

    # Start worker thread
    worker = threading.Thread(target=traffic_worker, args=(state, data_pool, API_URL), daemon=True)
    worker.start()

    # Setup terminal for Unix
    old_settings = None
    if sys.platform != 'win32':
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    # Enable ANSI escape codes on Windows
    if sys.platform == 'win32':
        os.system('')  # Enables ANSI escape sequences in Windows terminal

    clear_screen()

    try:
        while state.running:
            # Render display
            render_display(state)

            # Check for keypress
            key = get_key()
            if key:
                if key == 'q' or key == 'Q':
                    state.running = False
                elif key == ' ':
                    with state.lock:
                        state.paused = not state.paused
                elif key == '0':
                    with state.lock:
                        state.mode = 'normal'
                elif key in ATTACK_SCENARIOS:
                    with state.lock:
                        state.mode = key
                elif key == '+' or key == '=':
                    with state.lock:
                        state.speed = min(state.speed + 1.0, 20.0)
                elif key == '-' or key == '_':
                    with state.lock:
                        state.speed = max(state.speed - 1.0, 0.5)

            time.sleep(0.2)  # Display refresh rate

    except KeyboardInterrupt:
        state.running = False
    finally:
        # Restore terminal
        if old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # Final stats
        clear_screen()
        elapsed = time.time() - state.start_time
        print(f"\n{BOLD}{'=' * 60}{RESET}")
        print(f"{BOLD}  DEMO SESSION SUMMARY{RESET}")
        print(f"{'=' * 60}")
        print(f"  Duration:          {time.strftime('%M:%S', time.gmtime(elapsed))}")
        print(f"  Total flows sent:  {state.total_sent:,}")
        print(f"  Attacks sent:      {state.total_attacks_sent:,}")
        print(f"  Attacks detected:  {state.total_attacks_detected:,}")
        if state.total_sent > 0:
            print(f"  Overall accuracy:  {state.total_correct / state.total_sent * 100:.1f}%")
        if state.total_attacks_sent > 0:
            print(f"  Detection rate:    {state.total_attacks_detected / state.total_attacks_sent * 100:.1f}%")
        print(f"  Errors:            {state.errors}")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
