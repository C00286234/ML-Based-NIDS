"""
NIDS Launcher
=============
Starts both the ML API and NIDS Engine, saving all output to timestamped
session logs under logs/sessions/.

Usage:
    python run_nids.py                  # WiFi interface (default)
    python run_nids.py -i Ethernet      # Specify interface
    python run_nids.py --api-only       # Start API only
    python run_nids.py --engine-only    # Start engine only (API must be running)
"""

import os
import sys
import time
import threading
import subprocess
import argparse
from datetime import datetime

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))


def pipe_output(proc, log_file, label, show_console=True):
    """Read process stdout line-by-line, write to log file and optionally console."""
    with open(log_file, 'w', encoding='utf-8') as f:
        for line in iter(proc.stdout.readline, b''):
            decoded = line.decode('utf-8', errors='replace')
            f.write(decoded)
            f.flush()
            if show_console:
                sys.stdout.write(decoded)
                sys.stdout.flush()
    proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description="NIDS Launcher — starts API + Engine")
    parser.add_argument("-i", "--interface", default=None, help="Network interface (default: from config)")
    parser.add_argument("--api-only", action="store_true", help="Start API only")
    parser.add_argument("--engine-only", action="store_true", help="Start engine only (API must already be running)")
    parser.add_argument("--replay", metavar="CSV", default=None,
                        help="Replay CIC-IDS2017 flows from CSV instead of live capture")
    parser.add_argument("--speed", type=float, default=3.0,
                        help="Flows per second in replay mode (default: 3.0)")
    parser.add_argument("--attack-ratio", type=float, default=0.3,
                        help="Proportion of attack traffic in replay mode (default: 0.3)")
    args = parser.parse_args()

    # Session directory
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(ROOT, "logs", "sessions", session_ts)
    os.makedirs(session_dir, exist_ok=True)

    print("=" * 60)
    print("    NIDS LAUNCHER")
    print("=" * 60)
    print(f"  Session:  {session_ts}")
    print(f"  Logs:     {session_dir}")
    print("=" * 60)

    processes = []
    threads = []

    try:
        # --- Start API ---
        if not args.engine_only:
            api_log = os.path.join(session_dir, "api_output.log")

            print(f"\n[1] Starting ML API (port 8000)...")
            api_proc = subprocess.Popen(
                [sys.executable, "-u", os.path.join(ROOT, "api", "app.py")],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
            )
            processes.append(("API", api_proc))
            print(f"    PID: {api_proc.pid}  |  Log: api_output.log")

            # Pipe API output to log file (no console — too noisy)
            api_thread = threading.Thread(
                target=pipe_output,
                args=(api_proc, api_log, "API", False),
                daemon=True
            )
            api_thread.start()
            threads.append(api_thread)

            # Wait for API to be ready
            print("    Waiting for API startup...", end="", flush=True)
            import requests
            for i in range(30):
                time.sleep(1)
                try:
                    r = requests.get("http://localhost:8000/health", timeout=2)
                    if r.status_code == 200:
                        print(f" ready! ({i+1}s)")
                        break
                except Exception:
                    print(".", end="", flush=True)
            else:
                print("\n    WARNING: API did not respond in 30s. Check api_output.log")

        # --- Start Engine ---
        if not args.api_only:
            engine_log = os.path.join(session_dir, "engine_output.log")

            engine_cmd = [sys.executable, "-u", os.path.join(ROOT, "src", "nids_engine.py")]
            if args.interface:
                engine_cmd += ["--interface", args.interface]
            if args.replay:
                engine_cmd += ["--replay", args.replay,
                               "--speed", str(args.speed),
                               "--attack-ratio", str(args.attack_ratio)]

            mode_label = "NIDS Engine (REPLAY)" if args.replay else "NIDS Engine"
            print(f"\n[2] Starting {mode_label}...")
            engine_proc = subprocess.Popen(
                engine_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
            )
            processes.append(("Engine", engine_proc))
            print(f"    PID: {engine_proc.pid}  |  Log: engine_output.log")

            # Pipe engine output to both console and log file
            engine_thread = threading.Thread(
                target=pipe_output,
                args=(engine_proc, engine_log, "Engine", True),
                daemon=True
            )
            engine_thread.start()
            threads.append(engine_thread)

        print("\n" + "=" * 60)
        print("  NIDS is running. Press Ctrl+C to stop.")
        print(f"  Session logs: {session_dir}")
        print("=" * 60 + "\n")

        # Wait until a process exits or Ctrl+C
        while True:
            time.sleep(1)
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n{name} exited with code {proc.returncode}")
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        # Terminate all processes
        for name, proc in processes:
            if proc.poll() is None:
                print(f"  Stopping {name} (PID {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

        # Wait for output threads to flush
        for t in threads:
            t.join(timeout=5)

        # List session files
        print(f"\nSession logs saved to: {session_dir}")
        print("Files:")
        for f in sorted(os.listdir(session_dir)):
            size = os.path.getsize(os.path.join(session_dir, f))
            print(f"  {f:30s} {size:>8,} bytes")
        print("\nDone.")


if __name__ == "__main__":
    main()
