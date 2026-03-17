"""
Network Interface Verifier
===========================
Lists all Scapy-visible interfaces with their IPs and optionally
captures a few packets to confirm traffic is visible on a given interface.

Author: C00286234

Use this to find the correct VirtualBox Host-Only adapter for Kali VM testing.

Usage:
    python testing/verify_interface.py                  # List all interfaces
    python testing/verify_interface.py --test "Ethernet 4"   # Test specific interface
    python testing/verify_interface.py --test "Ethernet 4" --count 10  # Capture 10 packets
"""

import argparse
import sys
import time

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


# ==============================================================================
# INTERFACE LISTING
# ==============================================================================

def list_all_interfaces():
    """List all network interfaces visible to Scapy."""
    try:
        from scapy.all import conf
        import platform

        print_header("Available Network Interfaces")

        if platform.system() == "Windows":
            from scapy.arch.windows import get_windows_if_list
            interfaces = get_windows_if_list()

            for i, iface in enumerate(interfaces):
                name = iface.get("name", "Unknown")
                description = iface.get("description", "")
                ips = iface.get("ips", [])
                mac = iface.get("mac", "")

                # Filter for IPv4 addresses
                ipv4 = [ip for ip in ips if "." in str(ip)]
                ipv6 = [ip for ip in ips if ":" in str(ip) and "." not in str(ip)]

                # Highlight likely VirtualBox host-only adapters
                is_vbox = "virtualbox" in description.lower() or "host-only" in description.lower()
                is_hyper_v = "hyper-v" in description.lower()

                tag = ""
                if is_vbox:
                    tag = f" {GREEN}<< VirtualBox Host-Only{RESET}"
                elif is_hyper_v:
                    tag = f" {YELLOW}(Hyper-V){RESET}"

                print(f"\n  [{i}] {BOLD}{name}{RESET}{tag}")
                if description:
                    print(f"      Description: {description}")
                if ipv4:
                    print(f"      IPv4:        {', '.join(str(ip) for ip in ipv4)}")
                if mac:
                    print(f"      MAC:         {mac}")

            print(f"\n  Default interface: {BOLD}{conf.iface}{RESET}")
            return [iface.get("name", "") for iface in interfaces]

        else:
            # Linux/Mac
            from scapy.all import get_if_list, get_if_addr
            interfaces = get_if_list()

            for i, name in enumerate(interfaces):
                addr = get_if_addr(name)
                print(f"  [{i}] {BOLD}{name}{RESET}  ->  {addr or 'no IP'}")

            return interfaces

    except ImportError:
        print(f"{RED}ERROR: Scapy not installed. Run: pip install scapy{RESET}")
        sys.exit(1)


# ==============================================================================
# PACKET CAPTURE TEST
# ==============================================================================

def test_interface(interface_name, packet_count=5, timeout=10):
    """Capture packets on a specific interface to verify it works."""
    try:
        from scapy.all import sniff, IP

        print_header(f"Testing Interface: {interface_name}")
        print(f"  Capturing up to {packet_count} packets (timeout: {timeout}s)...")
        print(f"  {YELLOW}Tip: Generate traffic by pinging this machine from Kali{RESET}")
        print()

        captured = []

        def process_packet(pkt):
            if IP in pkt:
                src = pkt[IP].src
                dst = pkt[IP].dst
                proto = pkt[IP].proto
                size = len(pkt)

                proto_name = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(proto, str(proto))
                captured.append(pkt)

                print(f"    [{len(captured):>3}] {src:>15} -> {dst:<15}  "
                      f"{proto_name:<5}  {size:>5} bytes")

        start = time.time()
        sniff(
            iface=interface_name,
            prn=process_packet,
            filter="ip",
            count=packet_count,
            timeout=timeout,
            store=False
        )
        elapsed = time.time() - start

        print(f"\n  Captured {BOLD}{len(captured)}{RESET} packets in {elapsed:.1f}s")

        if len(captured) > 0:
            print(f"  {GREEN}Interface is working! Traffic is visible.{RESET}")

            # Show unique IPs
            ips = set()
            for pkt in captured:
                if IP in pkt:
                    ips.add(pkt[IP].src)
                    ips.add(pkt[IP].dst)
            print(f"  Unique IPs seen: {', '.join(sorted(ips))}")

            return True
        else:
            print(f"  {RED}No packets captured. Possible causes:{RESET}")
            print(f"    - Wrong interface name")
            print(f"    - No traffic on this interface (try pinging from Kali)")
            print(f"    - Promiscuous mode not supported")
            print(f"    - Run as Administrator")
            return False

    except PermissionError:
        print(f"  {RED}Permission denied. Run as Administrator.{RESET}")
        return False
    except Exception as e:
        print(f"  {RED}Error: {e}{RESET}")
        return False


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify network interfaces for NIDS Kali VM testing"
    )
    parser.add_argument("--test", default=None,
                        help="Test a specific interface by capturing packets")
    parser.add_argument("--count", type=int, default=5,
                        help="Number of packets to capture during test (default: 5)")
    parser.add_argument("--timeout", type=int, default=15,
                        help="Capture timeout in seconds (default: 15)")
    args = parser.parse_args()

    # Always list interfaces
    interfaces = list_all_interfaces()

    if args.test:
        # Test specific interface
        test_interface(args.test, args.count, args.timeout)
    else:
        print(f"\n  {YELLOW}To test an interface:{RESET}")
        print(f"    python testing/verify_interface.py --test \"<interface name>\"")
        print(f"\n  {YELLOW}For Kali VM testing, use the VirtualBox Host-Only adapter.{RESET}")
        print(f"  Then start the NIDS with:")
        print(f"    python run_nids.py -i \"<interface name>\"")


if __name__ == "__main__":
    main()
    print()
    input("Press Enter to exit...")
