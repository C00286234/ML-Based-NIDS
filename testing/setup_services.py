"""
Target Services Launcher
=========================
Starts all target services that the Kali VM will attack during NIDS testing.

Services:
    - Vulnerable webapp on port 8080 (Flask - XSS, SQLi, brute force targets)
    - FTP server on port 21 (pyftpdlib - for FTP brute force testing)

Prerequisites:
    pip install flask pyftpdlib

Usage:
    python testing/setup_services.py                    # Start all services
    python testing/setup_services.py --webapp-only      # Just the webapp
    python testing/setup_services.py --no-ftp           # Skip FTP
    python testing/setup_services.py --host-ip 192.168.56.1  # Bind to specific IP
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ==============================================================================
# FORMATTING
# ==============================================================================

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

PROJECT_ROOT = Path(__file__).parent.parent


def print_header(text):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")


# ==============================================================================
# SERVICE CHECKERS
# ==============================================================================

def check_port_available(port):
    """Check if a port is available."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def check_dependency(module_name, pip_name=None):
    """Check if a Python module is importable."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        print(f"  {RED}Missing: {module_name}{RESET}")
        print(f"    Install: pip install {pip_name or module_name}")
        return False


def check_ssh_available():
    """Check if Windows OpenSSH server is installed and running."""
    if sys.platform != "win32":
        return shutil.which("sshd") is not None

    # Check Windows OpenSSH
    try:
        result = subprocess.run(
            ["sc", "query", "sshd"],
            capture_output=True, text=True, timeout=5
        )
        if "RUNNING" in result.stdout:
            return True
        elif "STOPPED" in result.stdout:
            print(f"  {YELLOW}OpenSSH Server is installed but stopped.{RESET}")
            print(f"    Start it: net start sshd (as Administrator)")
            return False
        else:
            print(f"  {YELLOW}OpenSSH Server not installed.{RESET}")
            print(f"    Install: Settings > Apps > Optional Features > OpenSSH Server")
            return False
    except Exception:
        return False


# ==============================================================================
# SERVICE LAUNCHERS
# ==============================================================================

def start_webapp(host_ip, port=8080):
    """Start the vulnerable webapp."""
    webapp_path = Path(__file__).parent / "vulnerable_webapp.py"

    if not check_port_available(port):
        print(f"  {RED}Port {port} already in use!{RESET}")
        return None

    print(f"  Starting vulnerable webapp on {host_ip}:{port}...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-u", str(webapp_path),
         "--host", host_ip, "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait a moment and check it started
    time.sleep(2)
    if proc.poll() is not None:
        print(f"  {RED}Webapp failed to start!{RESET}")
        output = proc.stdout.read().decode("utf-8", errors="replace")
        print(f"  Output: {output[:200]}")
        return None

    print(f"  {GREEN}Webapp running on http://{host_ip}:{port}{RESET}")
    return proc


def start_ftp(host_ip, port=21):
    """Start a simple FTP server using pyftpdlib."""
    if not check_dependency("pyftpdlib"):
        return None

    if not check_port_available(port):
        print(f"  {RED}Port {port} already in use!{RESET}")
        # Try alternative port
        port = 2121
        if not check_port_available(port):
            print(f"  {RED}Port {port} also in use. Skipping FTP.{RESET}")
            return None
        print(f"  {YELLOW}Using alternative port {port}{RESET}")

    print(f"  Starting FTP server on {host_ip}:{port}...")

    # Create a temp directory for FTP
    ftp_dir = PROJECT_ROOT / "testing" / "ftp_root"
    ftp_dir.mkdir(exist_ok=True)
    (ftp_dir / "readme.txt").write_text("NIDS test FTP server.\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Use pyftpdlib with a test user
    ftp_script = f"""
import sys
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

authorizer = DummyAuthorizer()
authorizer.add_user("admin", "password123", r"{ftp_dir}", perm="elradfmw")
authorizer.add_user("user", "letmein", r"{ftp_dir}", perm="elr")
authorizer.add_anonymous(r"{ftp_dir}")

handler = FTPHandler
handler.authorizer = authorizer
handler.passive_ports = range(60000, 60100)
handler.banner = "NIDS Test FTP Server"

server = FTPServer(("{host_ip}", {port}), handler)
print(f"FTP server started on {host_ip}:{port}")
print(f"  Users: admin/password123, user/letmein, anonymous")
server.serve_forever()
"""

    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", ftp_script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    time.sleep(2)
    if proc.poll() is not None:
        output = proc.stdout.read().decode("utf-8", errors="replace")
        if "Permission" in output or "denied" in output:
            print(f"  {RED}FTP on port {port} requires admin privileges.{RESET}")
            print(f"  {YELLOW}Run as Administrator, or use --no-ftp{RESET}")
        else:
            print(f"  {RED}FTP failed to start: {output[:200]}{RESET}")
        return None

    print(f"  {GREEN}FTP running on {host_ip}:{port}{RESET}")
    print(f"    Users: admin/password123, user/letmein, anonymous")
    return proc


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Start target services for Kali VM attack testing"
    )
    parser.add_argument("--host-ip", default="0.0.0.0",
                        help="IP to bind services to (default: 0.0.0.0)")
    parser.add_argument("--webapp-port", type=int, default=8080,
                        help="Webapp port (default: 8080)")
    parser.add_argument("--webapp-only", action="store_true",
                        help="Only start the webapp")
    parser.add_argument("--no-ftp", action="store_true",
                        help="Skip FTP server")
    args = parser.parse_args()

    print_header("NIDS Target Services Launcher")

    # Check dependencies
    print(f"\n  {BOLD}Checking dependencies...{RESET}")
    deps_ok = check_dependency("flask", "flask")
    if not args.no_ftp and not args.webapp_only:
        deps_ok = check_dependency("pyftpdlib", "pyftpdlib") and deps_ok
    if not deps_ok:
        print(f"\n  {RED}Install missing dependencies first.{RESET}")
        sys.exit(1)
    print(f"  {GREEN}All dependencies available.{RESET}")

    # Check SSH status
    if not args.webapp_only:
        print(f"\n  {BOLD}Checking SSH server...{RESET}")
        ssh_ok = check_ssh_available()
        if ssh_ok:
            print(f"  {GREEN}OpenSSH Server is running (port 22){RESET}")
        else:
            print(f"  {YELLOW}SSH not available - hydra SSH tests will be skipped{RESET}")

    # Start services
    processes = []

    print(f"\n  {BOLD}Starting services...{RESET}")

    # Webapp
    webapp_proc = start_webapp(args.host_ip, args.webapp_port)
    if webapp_proc:
        processes.append(("Webapp", webapp_proc))

    # FTP
    if not args.no_ftp and not args.webapp_only:
        ftp_proc = start_ftp(args.host_ip)
        if ftp_proc:
            processes.append(("FTP", ftp_proc))

    if not processes:
        print(f"\n  {RED}No services started!{RESET}")
        sys.exit(1)

    # Summary
    print_header("Services Running")
    for name, proc in processes:
        print(f"  {GREEN}[OK]{RESET} {name} (PID: {proc.pid})")
    if not args.webapp_only and ssh_ok:
        print(f"  {GREEN}[OK]{RESET} SSH (Windows service)")

    print(f"\n  {BOLD}Target IP for Kali:{RESET} {args.host_ip}")
    print(f"  {YELLOW}Press Ctrl+C to stop all services{RESET}")

    # Wait for Ctrl+C
    try:
        while True:
            time.sleep(1)
            # Check processes are still alive
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n  {RED}{name} stopped unexpectedly (exit code: {proc.returncode}){RESET}")
    except KeyboardInterrupt:
        print(f"\n\n  Stopping services...")
        for name, proc in processes:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"  Stopped {name}")
        print(f"  {GREEN}All services stopped.{RESET}")


if __name__ == "__main__":
    main()
