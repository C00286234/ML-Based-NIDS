#!/usr/bin/env python3
"""
Inject synthetic attack traffic from CIC-IDS2017 CSV into the network.
Reconstructs packets from flow features and sends them to a network interface.
"""

import sys
import time
from random import choice, randint

import pandas as pd
from scapy.all import ICMP, IP, TCP, UDP, RandShort, send


def get_synthetic_ip(attack_type):
    """Generate realistic synthetic IPs based on attack type."""
    # Src: victim networks
    src_base = choice(['192.168.10', '192.168.1', '10.0.1'])
    src_ip = f"{src_base}.{randint(1, 254)}"

    # Dst: external/target networks
    if attack_type in ['Bot', 'Infiltration']:
        dst_ip = f"203.0.113.{randint(1, 254)}"  # Command & control
    elif attack_type == 'PortScan':
        dst_ip = f"192.168.10.{randint(1, 254)}"
    else:
        dst_ip = f"8.8.{randint(0, 255)}.{randint(1, 254)}"

    return src_ip, dst_ip

def inject_attacks(csv_file, interface, num_flows=100):
    """Inject attack flows from CSV as network packets."""
    print(f"Loading {csv_file}...")
    df = pd.read_parquet(csv_file) if str(csv_file).endswith(".parquet") else pd.read_csv(csv_file)

    # Filter for attacks only
    attacks = df[df['Label'] != 'BENIGN'].copy()
    print(f"Found {len(attacks)} attack flows in CSV")

    if len(attacks) == 0:
        print("No attacks found!")
        return

    # Sample if too many
    attacks = attacks.sample(min(num_flows, len(attacks)))
    print(f"Injecting {len(attacks)} attack packets to interface: {interface}\n")

    for idx, row in attacks.iterrows():
        attack_type = row['Label']
        src_port = int(row['Src Port']) if row['Src Port'] > 0 else RandShort()
        dst_port = int(row['Dst Port']) if row['Dst Port'] > 0 else RandShort()
        protocol = int(row['Protocol'])

        # Generate synthetic IPs
        src_ip, dst_ip = get_synthetic_ip(attack_type)

        # Extract features for packet construction
        rst_flags = int(row.get('RST Flag Count', 0))
        fin_flags = int(row.get('FIN Flag Count', 0))

        # Create packets based on protocol
        try:
            if protocol == 6:  # TCP
                # Send SYN
                pkt_syn = IP(src=src_ip, dst=dst_ip) / TCP(
                    sport=src_port,
                    dport=dst_port,
                    flags='S',
                    seq=randint(1000, 999999)
                )
                send(pkt_syn, iface=interface, verbose=False)
                time.sleep(0.005)

                # Send SYN-ACK (response)
                pkt_synack = IP(src=dst_ip, dst=src_ip) / TCP(
                    sport=dst_port,
                    dport=src_port,
                    flags='SA',
                    seq=randint(1000, 999999),
                    ack=1
                )
                send(pkt_synack, iface=interface, verbose=False)
                time.sleep(0.005)

                # Send payload packets (3-5 packets)
                for _ in range(randint(3, 5)):
                    pkt_data = IP(src=src_ip, dst=dst_ip) / TCP(
                        sport=src_port,
                        dport=dst_port,
                        flags='PA',
                        seq=randint(1000, 999999)
                    )
                    send(pkt_data, iface=interface, verbose=False)
                    time.sleep(0.003)

                # Send FIN or RST if indicated
                if fin_flags > 0:
                    pkt_fin = IP(src=src_ip, dst=dst_ip) / TCP(
                        sport=src_port,
                        dport=dst_port,
                        flags='F'
                    )
                    send(pkt_fin, iface=interface, verbose=False)
                elif rst_flags > 0:
                    pkt_rst = IP(src=src_ip, dst=dst_ip) / TCP(
                        sport=src_port,
                        dport=dst_port,
                        flags='R'
                    )
                    send(pkt_rst, iface=interface, verbose=False)

            elif protocol == 17:  # UDP
                # Send 3 UDP packets
                for _ in range(3):
                    pkt = IP(src=src_ip, dst=dst_ip) / UDP(sport=src_port, dport=dst_port)
                    send(pkt, iface=interface, verbose=False)
                    time.sleep(0.005)

            elif protocol == 1:  # ICMP
                # Send 2 ICMP packets
                for _ in range(2):
                    pkt = IP(src=src_ip, dst=dst_ip) / ICMP()
                    send(pkt, iface=interface, verbose=False)
                    time.sleep(0.005)

            # Status update every 10 flows
            if (idx + 1) % 10 == 0:
                print(f"✓ Injected {idx + 1}/{len(attacks)} flows ({attack_type})")

        except Exception as e:
            print(f"✗ Error injecting {attack_type} flow: {e}")
            continue

    print(f"\n✓ Injection complete! Sent {len(attacks)} attack flows.")
    print("Engine should detect attacks in: logs/sessions/*/attacks.log")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inject_attacks.py <csv_file> <interface> [num_flows]")
        print("Example: python inject_attacks.py data/processed/test.parquet WiFi 100")
        sys.exit(1)

    csv_file = sys.argv[1]
    interface = sys.argv[2]
    num_flows = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    inject_attacks(csv_file, interface, num_flows)
