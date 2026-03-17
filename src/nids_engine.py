"""
NIDS Engine - Main Orchestrator
================================
The main engine that coordinates packet capture, flow aggregation,
feature extraction, and ML prediction for network intrusion detection.

Author: C00286234
"""

# --- Imports: standard library, third-party, and local NIDS modules ---
import os
import sys
import time
import signal
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict

import yaml
import requests

from packet_capture import PacketCapture, list_interfaces
from flow_aggregator import FlowAggregator, Flow
from feature_extractor import FeatureExtractor
from alerting import AlertManager

# --- Logging configuration: console output format ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("nids.engine")


class NIDSEngine:
    """
    Network Intrusion Detection System Engine.

    Orchestrates the complete NIDS pipeline:
    1. Capture packets from network interface
    2. Aggregate packets into flows
    3. Extract 84 CIC-IDS2017 features
    4. Send to ML API for prediction
    5. Log and alert on detected attacks

    Example:
        engine = NIDSEngine("config/nids_config.yaml")
        engine.start()  # Blocks until Ctrl+C
    """

    # --- Class initialisation: load config, set up logging, create components ---
    def __init__(self, config_path: str):
        """
        Initialize the NIDS engine.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Initialize components
        self._init_logging()
        self._init_components()

        # Control flags
        self._running = False
        self._stop_event = threading.Event()

        # Thread pool for non-blocking API predictions
        # This prevents slow API calls from blocking packet ingestion
        self._prediction_pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="nids-predict"
        )
        self._pending_futures = []

        # Statistics
        self._stats = {
            "start_time": None,
            "flows_processed": 0,
            "predictions_made": 0,
            "attacks_detected": 0,
            "api_errors": 0,
            "attack_distribution": defaultdict(int)
        }
        self._stats_lock = threading.Lock()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("NIDS Engine initialized")

    # --- Config loading: read YAML file or fall back to defaults ---
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "network": {
                "interface": None,
                "promiscuous": True,
                "bpf_filter": "ip"
            },
            "flow": {
                "timeout": 120.0,
                "max_packets": 1000,
                "max_flows": 100000
            },
            "api": {
                "url": "http://localhost:8000",
                "predict_endpoint": "/predict",
                "timeout": 10.0,
                "retry_count": 3
            },
            "alerts": {
                "enabled": True,
                "confidence_threshold": 0.8,
                "log_only": True
            },
            "logging": {
                "level": "INFO",
                "file": "logs/nids/nids_engine.log"
            },
            "performance": {
                "packet_batch_size": 100,
                "cleanup_interval": 30,
                "stats_interval": 10
            }
        }

    # --- Logging setup: session logs, rolling logs, prediction and attack loggers ---
    def _init_logging(self) -> None:
        """Initialize file logging based on config."""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")

        # Resolve log paths relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Create session-based log directory: logs/sessions/YYYYMMDD_HHMMSS/
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join(project_root, "logs", "sessions", session_ts)
        os.makedirs(self._session_dir, exist_ok=True)

        # Also keep the rolling log for quick access
        rolling_log_dir = os.path.join(project_root, "logs", "nids")
        os.makedirs(rolling_log_dir, exist_ok=True)

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Session log (one file per run)
        session_handler = logging.FileHandler(
            os.path.join(self._session_dir, "engine.log"), encoding='utf-8'
        )
        session_handler.setLevel(getattr(logging, log_level))
        session_handler.setFormatter(formatter)
        logging.getLogger().addHandler(session_handler)

        # Rolling log (appended across runs)
        rolling_handler = logging.FileHandler(
            os.path.join(rolling_log_dir, "nids_engine.log"), encoding='utf-8'
        )
        rolling_handler.setLevel(getattr(logging, log_level))
        rolling_handler.setFormatter(formatter)
        logging.getLogger().addHandler(rolling_handler)

        # Session prediction log (every prediction in this run)
        self._prediction_logger = logging.getLogger("nids.predictions")
        pred_handler = logging.FileHandler(
            os.path.join(self._session_dir, "predictions.log"), encoding='utf-8'
        )
        pred_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self._prediction_logger.addHandler(pred_handler)

        # Session attack log
        self._attack_logger = logging.getLogger("nids.attacks")
        attack_handler = logging.FileHandler(
            os.path.join(self._session_dir, "attacks.log"), encoding='utf-8'
        )
        attack_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self._attack_logger.addHandler(attack_handler)

        # Write session start marker
        logger.info(f"=== SESSION START {session_ts} ===")
        logger.info(f"Session logs: {self._session_dir}")

    # --- Component initialisation: packet capture, flow aggregator, feature extractor, alerting ---
    def _init_components(self) -> None:
        """Initialize NIDS components."""
        net_config = self.config.get("network", {})
        flow_config = self.config.get("flow", {})

        # Packet capture
        self.packet_capture = PacketCapture(
            interface=net_config.get("interface"),
            promiscuous=net_config.get("promiscuous", True),
            bpf_filter=net_config.get("bpf_filter", "ip")
        )

        # Flow aggregator
        self.flow_aggregator = FlowAggregator(
            flow_timeout=flow_config.get("timeout", 120.0),
            max_packets_per_flow=flow_config.get("max_packets", 1000),
            max_flows=flow_config.get("max_flows", 100000)
        )

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Alert manager (email + Slack)
        self.alert_manager = AlertManager(self.config)

        logger.info("All components initialized")

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    # --- API health check: verify ML prediction service is reachable and models loaded ---
    def _check_api_health(self) -> bool:
        """Check if the ML API is reachable."""
        api_config = self.config.get("api", {})
        url = api_config.get("url", "http://localhost:8000")

        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("models_loaded"):
                    logger.info(f"API is healthy and models are loaded")
                    return True
                else:
                    logger.warning("API is running but models not loaded")
                    return False
            else:
                logger.warning(f"API health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to API: {e}")
            return False

    # --- Prediction pipeline: send features to ML API and handle retries ---
    def _predict_flow(
        self,
        features: Dict[str, float],
        flow_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Send flow features to ML API for prediction.

        Args:
            features: Dictionary of 84 features
            flow_id: Optional flow identifier for log tracing

        Returns:
            Prediction response or None on error
        """
        api_config = self.config.get("api", {})
        url = api_config.get("url", "http://localhost:8000")
        endpoint = api_config.get("predict_endpoint", "/predict")
        timeout = api_config.get("timeout", 10.0)
        retry_count = api_config.get("retry_count", 3)

        payload = {"features": features}
        if flow_id:
            payload["flow_id"] = flow_id

        for attempt in range(retry_count):
            try:
                response = requests.post(
                    f"{url}{endpoint}",
                    json=payload,
                    timeout=timeout
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"API returned {response.status_code}: {response.text[:100]}")

            except requests.exceptions.Timeout:
                logger.warning(f"API timeout (attempt {attempt + 1}/{retry_count})")
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")

            if attempt < retry_count - 1:
                time.sleep(0.5)

        with self._stats_lock:
            self._stats["api_errors"] += 1
        return None

    # --- Prediction handling: log results, update stats, trigger alerts for attacks ---
    def _handle_prediction(
        self,
        flow: Flow,
        features: Dict[str, float],
        prediction: Dict[str, Any]
    ) -> None:
        """
        Handle a prediction result.

        Args:
            flow: The original flow
            features: Extracted features
            prediction: API prediction response
        """
        alert_config = self.config.get("alerts", {})
        threshold = alert_config.get("confidence_threshold", 0.8)

        pred_label = prediction.get("prediction", "UNKNOWN")
        confidence = prediction.get("confidence", 0.0)
        is_attack = prediction.get("is_attack", False)

        # Map protocol number to name for readability
        proto_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
        proto_name = proto_map.get(flow.protocol, str(flow.protocol))

        # --- Real-time console output for every prediction ---
        flow_summary = (
            f"{flow.src_ip}:{flow.src_port} -> "
            f"{flow.dst_ip}:{flow.dst_port} ({proto_name})"
        )

        if is_attack:
            # Red/bold for attacks
            print(
                f"  [!] ATTACK  {pred_label:<28} "
                f"conf={confidence:.0%}  {flow_summary}  "
                f"pkts={flow.total_packets}"
            )
        else:
            # Normal for benign
            print(
                f"  [ ] BENIGN  {'':<28} "
                f"conf={confidence:.0%}  {flow_summary}  "
                f"pkts={flow.total_packets}"
            )

        # --- Log every prediction to session file ---
        self._prediction_logger.info(
            f"{'ATTACK' if is_attack else 'BENIGN'} | "
            f"{pred_label:<28} | conf={confidence:.4f} | "
            f"{flow_summary} | pkts={flow.total_packets}"
        )

        # --- Update stats (thread-safe) ---
        with self._stats_lock:
            self._stats["predictions_made"] += 1

        # --- Attack-specific handling ---
        if is_attack and confidence >= threshold:
            with self._stats_lock:
                self._stats["attacks_detected"] += 1
                self._stats["attack_distribution"][pred_label] += 1

            attack_msg = (
                f"ATTACK DETECTED | "
                f"Type: {pred_label} | "
                f"Confidence: {confidence:.2%} | "
                f"Flow: {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} | "
                f"Protocol: {proto_name} | "
                f"Packets: {flow.total_packets}"
            )

            self._attack_logger.warning(attack_msg)
            logger.warning(attack_msg)

            # Send alerts (email + Slack)
            self.alert_manager.send_attack_alert(
                attack_type=pred_label,
                confidence=confidence,
                source_ip=flow.src_ip,
                destination_ip=flow.dst_ip,
                source_port=flow.src_port,
                destination_port=flow.dst_port,
                protocol=flow.protocol,
                features=features,
                flow_id=flow.flow_id,
            )

            # Log top contributing features for attacks
            top_features = prediction.get("top_features", [])
            if top_features:
                feat_str = ", ".join(
                    f"{f['feature']}={f['contribution']:+.3f}"
                    for f in top_features[:3]
                )
                logger.info(f"  SHAP: {feat_str}")

    def _predict_and_handle(self, flow: Flow, features: Dict[str, float]) -> None:
        """
        Predict and handle a single flow. Runs in a thread pool worker.

        Args:
            flow: The original flow
            features: Extracted 84 features
        """
        try:
            prediction = self._predict_flow(features, flow_id=flow.flow_id)
            if prediction:
                self._handle_prediction(flow, features, prediction)
        except Exception as e:
            logger.error(f"Error predicting flow {flow.flow_id}: {e}")

    def _collect_completed_predictions(self) -> None:
        """Clean up finished prediction futures."""
        still_pending = []
        for future in self._pending_futures:
            if future.done():
                # Retrieve exception if any (prevents silent failures)
                exc = future.exception()
                if exc:
                    logger.error(f"Prediction thread error: {exc}")
            else:
                still_pending.append(future)
        self._pending_futures = still_pending

    # --- Flow processing: extract features and submit predictions to thread pool ---
    def _process_flows(self) -> int:
        """
        Process all completed flows.

        Feature extraction happens here (fast, CPU-bound).
        API predictions are submitted to the thread pool (slow, I/O-bound).

        Returns:
            Number of flows processed
        """
        completed_flows = self.flow_aggregator.get_completed_flows()

        for flow in completed_flows:
            try:
                # Extract features (fast - runs inline)
                features = self.feature_extractor.extract_features(flow)

                # Validate features
                if not self.feature_extractor.validate_features(features):
                    logger.warning(f"Invalid features for flow {flow.flow_id}")
                    continue

                # Submit prediction to thread pool (non-blocking)
                future = self._prediction_pool.submit(
                    self._predict_and_handle, flow, features
                )
                self._pending_futures.append(future)

                with self._stats_lock:
                    self._stats["flows_processed"] += 1

            except Exception as e:
                logger.error(f"Error processing flow {flow.flow_id}: {e}")

        # Clean up completed futures
        self._collect_completed_predictions()

        return len(completed_flows)

    # --- Statistics: display capture, flow, and prediction metrics ---
    def _print_statistics(self) -> None:
        """Print current statistics."""
        uptime = time.time() - self._stats["start_time"]

        capture_stats = self.packet_capture.get_statistics()
        flow_stats = self.flow_aggregator.get_statistics()

        print("\n" + "=" * 70)
        print(f"NIDS Engine Statistics (uptime: {uptime:.1f}s)")
        print("=" * 70)

        total_pkts = capture_stats['packets_captured'] + capture_stats['packets_dropped']
        drop_pct = (capture_stats['packets_dropped'] / total_pkts * 100) if total_pkts > 0 else 0

        print(f"\nPacket Capture:")
        print(f"  Captured:    {capture_stats['packets_captured']:,}")
        print(f"  Dropped:     {capture_stats['packets_dropped']:,} ({drop_pct:.1f}%)")
        print(f"  Queue size:  {capture_stats['queue_size']:,}")
        print(f"  Rate:        {capture_stats['packets_per_second']:.1f} pps")

        print(f"\nFlow Aggregation:")
        print(f"  Active flows:    {flow_stats['active_flows']:,}")
        print(f"  Total created:   {flow_stats['total_flows_created']:,}")
        print(f"  Total completed: {flow_stats['total_flows_completed']:,}")

        print(f"\nPredictions:")
        print(f"  Flows processed: {self._stats['flows_processed']:,}")
        print(f"  Predictions:     {self._stats['predictions_made']:,}")
        print(f"  Attacks found:   {self._stats['attacks_detected']:,}")
        print(f"  API errors:      {self._stats['api_errors']:,}")
        print(f"  Pending:         {len(self._pending_futures):,}")

        if self._stats["attack_distribution"]:
            print(f"\nAttack Distribution:")
            for attack_type, count in sorted(
                self._stats["attack_distribution"].items(),
                key=lambda x: -x[1]
            ):
                print(f"  {attack_type}: {count}")

        print("=" * 70)

    # --- Main loop: ingest packets, aggregate flows, run predictions, periodic cleanup ---
    def _main_loop(self) -> None:
        """Main processing loop."""
        perf_config = self.config.get("performance", {})
        batch_size = perf_config.get("packet_batch_size", 100)
        cleanup_interval = perf_config.get("cleanup_interval", 30)
        stats_interval = perf_config.get("stats_interval", 10)

        last_cleanup = time.time()
        last_stats = time.time()

        logger.info("Main processing loop started")

        while self._running and not self._stop_event.is_set():
            try:
                # Get captured packets
                packets = self.packet_capture.get_packets(max_count=batch_size)

                # Process packets into flows
                for packet in packets:
                    self.flow_aggregator.process_packet(packet)

                # Process completed flows
                flows_processed = self._process_flows()

                # Periodic cleanup
                current_time = time.time()
                if current_time - last_cleanup > cleanup_interval:
                    self.flow_aggregator.cleanup_expired_flows()
                    last_cleanup = current_time

                # Periodic statistics
                if current_time - last_stats > stats_interval:
                    self._print_statistics()
                    last_stats = current_time

                # Small sleep to prevent CPU spinning
                if not packets:
                    time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)

    # --- Start/stop lifecycle: launch capture, enter main loop, handle shutdown ---
    def start(self) -> None:
        """
        Start the NIDS engine.

        This method blocks until stop() is called or Ctrl+C is pressed.
        """
        if self._running:
            logger.warning("NIDS Engine is already running")
            return

        print("\n" + "=" * 70)
        print("    NETWORK INTRUSION DETECTION SYSTEM")
        print("=" * 70)

        # Check API health
        print("\nChecking ML API connection...")
        if not self._check_api_health():
            print("WARNING: ML API is not available!")
            print("Please start the API first: cd api && python -m uvicorn app:app --port 8000")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                return

        # Show network interfaces
        print("\nAvailable network interfaces:")
        interfaces = list_interfaces()
        for i, iface in enumerate(interfaces):
            print(f"  {i}: {iface}")

        interface = self.config.get("network", {}).get("interface")
        print(f"\nUsing interface: {interface or 'default'}")

        # Start components
        print("\nStarting NIDS components...")
        self._running = True
        self._stats["start_time"] = time.time()

        try:
            # Start packet capture
            self.packet_capture.start()
            logger.info("Packet capture started")

            print("\n" + "=" * 70)
            print("NIDS Engine is running. Press Ctrl+C to stop.")
            print("=" * 70 + "\n")

            # Run main loop
            self._main_loop()

        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the NIDS engine gracefully."""
        if not self._running:
            return

        logger.info("Stopping NIDS Engine...")
        self._running = False
        self._stop_event.set()

        # Stop packet capture
        self.packet_capture.stop()

        # Wait for in-flight predictions to finish
        if self._pending_futures:
            logger.info(f"Waiting for {len(self._pending_futures)} pending predictions...")
            for future in self._pending_futures:
                try:
                    future.result(timeout=10)
                except Exception as e:
                    logger.error(f"Pending prediction error: {e}")
            self._pending_futures.clear()

        # Process remaining flows
        remaining_flows = self.flow_aggregator.force_complete_all()
        logger.info(f"Processing {len(remaining_flows)} remaining flows...")

        for flow in remaining_flows:
            try:
                features = self.feature_extractor.extract_features(flow)
                prediction = self._predict_flow(features, flow_id=flow.flow_id)
                if prediction:
                    self._handle_prediction(flow, features, prediction)
                with self._stats_lock:
                    self._stats["flows_processed"] += 1
            except Exception as e:
                logger.error(f"Error processing final flow: {e}")

        # Shut down thread pool
        self._prediction_pool.shutdown(wait=True, cancel_futures=False)

        # Print final statistics
        self._print_statistics()

        # Write session summary to file
        self._save_session_summary()

        logger.info("=== SESSION END ===")
        logger.info("NIDS Engine stopped")
        print("\nNIDS Engine stopped.")

    # --- Session summary: write human-readable report to session log directory ---
    def _save_session_summary(self) -> None:
        """Save a human-readable session summary to the session log directory."""
        if not hasattr(self, '_session_dir'):
            return

        uptime = time.time() - self._stats["start_time"] if self._stats["start_time"] else 0
        capture_stats = self.packet_capture.get_statistics()

        summary_path = os.path.join(self._session_dir, "summary.txt")
        try:
            with open(summary_path, 'w') as f:
                f.write("NIDS Session Summary\n")
                f.write("=" * 60 + "\n")
                f.write(f"Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Uptime:   {uptime:.1f}s ({uptime/60:.1f} min)\n")
                f.write(f"Interface: {capture_stats.get('interface', 'N/A')}\n\n")

                f.write("Packet Capture\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Captured:  {capture_stats['packets_captured']:,}\n")
                f.write(f"  Dropped:   {capture_stats['packets_dropped']:,}\n")
                f.write(f"  Rate:      {capture_stats['packets_per_second']:.1f} pps\n\n")

                f.write("Predictions\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Flows processed: {self._stats['flows_processed']:,}\n")
                f.write(f"  Predictions:     {self._stats['predictions_made']:,}\n")
                f.write(f"  Attacks found:   {self._stats['attacks_detected']:,}\n")
                f.write(f"  API errors:      {self._stats['api_errors']:,}\n\n")

                if self._stats["attack_distribution"]:
                    f.write("Attack Distribution\n")
                    f.write("-" * 40 + "\n")
                    for attack_type, count in sorted(
                        self._stats["attack_distribution"].items(),
                        key=lambda x: -x[1]
                    ):
                        f.write(f"  {attack_type}: {count}\n")

            logger.info(f"Session summary saved: {summary_path}")
            print(f"\nSession logs saved to: {self._session_dir}")
        except Exception as e:
            logger.error(f"Failed to save session summary: {e}")


# --- CLI entry point: parse arguments, create engine, and start ---
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Network Intrusion Detection System Engine"
    )
    # Default config path relative to project root (one level up from src/)
    default_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "nids_config.yaml"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=default_config,
        help="Path to configuration file (default: config/nids_config.yaml)"
    )
    parser.add_argument(
        "--interface", "-i",
        help="Network interface to capture from"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="ML API URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--list-interfaces",
        action="store_true",
        help="List available network interfaces and exit"
    )

    args = parser.parse_args()

    if args.list_interfaces:
        print("Available network interfaces:")
        for i, iface in enumerate(list_interfaces()):
            print(f"  {i}: {iface}")
        return

    # Create engine
    engine = NIDSEngine(args.config)

    # Override config with command line arguments
    if args.interface:
        engine.config["network"]["interface"] = args.interface
        engine.packet_capture.interface = args.interface  # Fix: update already-created PacketCapture
    if args.api_url:
        engine.config["api"]["url"] = args.api_url

    # Start engine
    engine.start()


if __name__ == "__main__":
    main()
