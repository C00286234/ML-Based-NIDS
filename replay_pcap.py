from scapy.all import rdpcap, send
import sys
import time

if len(sys.argv) < 2:
      print("Usage: python replay_pcap.py <pcap_file> [interface]")
      sys.exit(1)

pcap_file = sys.argv[1]
interface = sys.argv[2] if len(sys.argv) > 2 else None

print(f"Loading {pcap_file}...")
packets = rdpcap(pcap_file)
print(f"Found {len(packets)} packets")

print(f"Replaying to interface: {interface or 'default'}")
for i, pkt in enumerate(packets):
      send(pkt, iface=interface, verbose=False)
      if (i + 1) % 100 == 0:
          print(f"Sent {i + 1}/{len(packets)} packets...")
      time.sleep(0.001)  # 1ms delay between packets

print("Replay complete!")