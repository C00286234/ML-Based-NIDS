"""
NIDS Alerting Module
====================
Provides email and Slack alerting for detected attacks.

Features:
- SMTP email support (Gmail, Outlook, custom SMTP)
- Slack webhook notifications
- Configurable thresholds
- Cooldown period to prevent alert flooding
- HTML email templates
- Attack severity levels

Configuration (in nids_config.yaml):
    alerting:
      email:
        enabled: true
        smtp_server: "smtp.gmail.com"
        smtp_port: 587
        use_tls: true
        sender: "nids-alerts@example.com"
        password: "${NIDS_EMAIL_PASSWORD}"
        recipients: ["security-team@example.com"]
        min_confidence: 0.9
        cooldown_minutes: 5
        severity_threshold: "medium"
      slack:
        enabled: true
        webhook_url: "${NIDS_SLACK_WEBHOOK}"
        channel: "#nids-alerts"
        min_confidence: 0.8
        cooldown_minutes: 2
        severity_threshold: "medium"

Usage:
    from alerting import AlertManager

    alert_manager = AlertManager(config)
    alert_manager.send_attack_alert(
        attack_type="DDoS",
        confidence=0.95,
        source_ip="192.168.1.100",
        destination_ip="10.0.0.1",
        features={"Flow Duration": 1000000, ...}
    )
"""

import logging
import os
import smtplib
import ssl
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import requests as http_requests

logger = logging.getLogger("nids.alerting")

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Attack severity levels
SEVERITY_LEVELS = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4
}

# Map attack types to severity
from nids.features import ATTACK_SEVERITY  # noqa: E402  (re-exported for callers)

# ==============================================================================
# DATA CLASSES
# ==============================================================================

def _resolve_env_var(value: str) -> str:
    """Substitute ${VAR_NAME} placeholders with environment variable values."""
    if value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    return value


@dataclass
class AlertConfig:
    """Email alerting configuration."""
    enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    sender: str = ""
    password: str = ""
    recipients: list[str] = field(default_factory=list)
    min_confidence: float = 0.9
    cooldown_minutes: int = 5
    severity_threshold: str = "medium"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "AlertConfig":
        """Create AlertConfig from dictionary."""
        email_config = config.get("alerting", {}).get("email", {})

        password = _resolve_env_var(email_config.get("password", ""))

        return cls(
            enabled=email_config.get("enabled", False),
            smtp_server=email_config.get("smtp_server", "smtp.gmail.com"),
            smtp_port=email_config.get("smtp_port", 587),
            use_tls=email_config.get("use_tls", True),
            sender=email_config.get("sender", ""),
            password=password,
            recipients=email_config.get("recipients", []),
            min_confidence=email_config.get("min_confidence", 0.9),
            cooldown_minutes=email_config.get("cooldown_minutes", 5),
            severity_threshold=email_config.get("severity_threshold", "medium"),
        )


@dataclass
class SlackConfig:
    """Slack webhook alerting configuration."""
    enabled: bool = False
    webhook_url: str = ""
    channel: str = "#nids-alerts"
    min_confidence: float = 0.8
    cooldown_minutes: int = 2
    severity_threshold: str = "medium"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "SlackConfig":
        """Create SlackConfig from dictionary."""
        slack_config = config.get("alerting", {}).get("slack", {})

        webhook_url = _resolve_env_var(slack_config.get("webhook_url", ""))

        return cls(
            enabled=slack_config.get("enabled", False),
            webhook_url=webhook_url,
            channel=slack_config.get("channel", "#nids-alerts"),
            min_confidence=slack_config.get("min_confidence", 0.8),
            cooldown_minutes=slack_config.get("cooldown_minutes", 2),
            severity_threshold=slack_config.get("severity_threshold", "medium"),
        )


@dataclass
class AttackAlert:
    """Represents an attack alert."""
    attack_type: str
    confidence: float
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: int
    timestamp: datetime
    severity: str
    features: dict[str, float] = field(default_factory=dict)
    flow_id: str | None = None


# ==============================================================================
# ALERT MANAGER
# ==============================================================================

class AlertManager:
    """
    Manages attack alerting via email and Slack.

    Handles:
    - Email sending via SMTP
    - Slack webhook notifications
    - Cooldown periods to prevent flooding
    - Severity filtering
    - Alert statistics
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize AlertManager.

        Args:
            config: Configuration dictionary (from nids_config.yaml)
        """
        self.email_config = AlertConfig.from_dict(config)
        self.slack_config = SlackConfig.from_dict(config)

        # Backwards compatibility
        self.config = self.email_config

        # Separate cooldown tracking per channel
        self._last_email_time: dict[str, datetime] = defaultdict(lambda: datetime.min)
        self._last_slack_time: dict[str, datetime] = defaultdict(lambda: datetime.min)
        self._alert_counts: dict[str, int] = defaultdict(int)
        self._total_alerts_sent = 0

        # Validate email config
        if self.email_config.enabled:
            if not self.email_config.sender or not self.email_config.password:
                logger.warning("Email alerting enabled but sender/password not configured")
                self.email_config.enabled = False
            elif not self.email_config.recipients:
                logger.warning("Email alerting enabled but no recipients configured")
                self.email_config.enabled = False
            else:
                logger.info(f"Email alerting enabled: {len(self.email_config.recipients)} recipient(s)")
        else:
            logger.info("Email alerting disabled")

        # Validate Slack config
        if self.slack_config.enabled:
            if not self.slack_config.webhook_url:
                logger.warning("Slack alerting enabled but webhook URL not configured")
                self.slack_config.enabled = False
            else:
                logger.info(f"Slack alerting enabled: {self.slack_config.channel}")
        else:
            logger.info("Slack alerting disabled")

    def _should_send(self, attack_type: str, confidence: float,
                     config, last_times: dict[str, datetime]) -> bool:
        """Check if an alert should be sent for a given channel."""
        if not config.enabled:
            return False
        if confidence < config.min_confidence:
            return False

        # Check severity threshold
        attack_severity = ATTACK_SEVERITY.get(attack_type, "medium")
        min_severity = SEVERITY_LEVELS.get(config.severity_threshold, 2)
        if SEVERITY_LEVELS.get(attack_severity, 2) < min_severity:
            return False

        # Check cooldown
        last_alert = last_times[attack_type]
        cooldown = timedelta(minutes=config.cooldown_minutes)
        if datetime.now() - last_alert < cooldown:
            return False

        return True

    def should_send_alert(self, attack_type: str, confidence: float) -> bool:
        """Check if any channel should send an alert."""
        return (
            self._should_send(attack_type, confidence, self.email_config, self._last_email_time) or
            self._should_send(attack_type, confidence, self.slack_config, self._last_slack_time)
        )

    def send_attack_alert(
        self,
        attack_type: str,
        confidence: float,
        source_ip: str,
        destination_ip: str,
        source_port: int = 0,
        destination_port: int = 0,
        protocol: int = 6,
        features: dict[str, float] | None = None,
        flow_id: str | None = None,
    ) -> bool:
        """
        Send attack alerts via all enabled channels (email + Slack).

        Returns:
            True if at least one alert was sent successfully
        """
        alert = AttackAlert(
            attack_type=attack_type,
            confidence=confidence,
            source_ip=source_ip,
            destination_ip=destination_ip,
            source_port=source_port,
            destination_port=destination_port,
            protocol=protocol,
            timestamp=datetime.now(),
            severity=ATTACK_SEVERITY.get(attack_type, "medium"),
            features=features or {},
            flow_id=flow_id,
        )

        sent_any = False

        # Email
        if self._should_send(attack_type, confidence, self.email_config, self._last_email_time):
            try:
                self._send_email(alert)
                self._last_email_time[attack_type] = datetime.now()
                sent_any = True
                logger.info(f"Email alert sent for {attack_type} from {source_ip}")
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")

        # Slack
        if self._should_send(attack_type, confidence, self.slack_config, self._last_slack_time):
            try:
                self._send_slack(alert)
                self._last_slack_time[attack_type] = datetime.now()
                sent_any = True
                logger.info(f"Slack alert sent for {attack_type} from {source_ip}")
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")

        if sent_any:
            self._alert_counts[attack_type] += 1
            self._total_alerts_sent += 1

        return sent_any

    def _send_email(self, alert: AttackAlert) -> None:
        """
        Send email via SMTP.

        Args:
            alert: AttackAlert object with alert details
        """
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = self._get_subject(alert)
        msg["From"] = self.config.sender
        msg["To"] = ", ".join(self.config.recipients)

        # Create plain text and HTML versions
        text_content = self._get_plain_text(alert)
        html_content = self._get_html_content(alert)

        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        # Send email
        context = ssl.create_default_context()

        if self.config.use_tls:
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.config.sender, self.config.password)
                server.sendmail(
                    self.config.sender,
                    self.config.recipients,
                    msg.as_string()
                )
        else:
            with smtplib.SMTP_SSL(
                self.config.smtp_server,
                self.config.smtp_port,
                context=context
            ) as server:
                server.login(self.config.sender, self.config.password)
                server.sendmail(
                    self.config.sender,
                    self.config.recipients,
                    msg.as_string()
                )

    def _send_slack(self, alert: AttackAlert) -> None:
        """
        Send alert via Slack webhook.

        Uses Slack Block Kit for rich formatting with severity colors.
        """
        severity_emoji = {
            "low": ":large_yellow_circle:",
            "medium": ":large_orange_circle:",
            "high": ":red_circle:",
            "critical": ":rotating_light:"
        }

        emoji = severity_emoji.get(alert.severity, ":warning:")
        proto_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
        proto_name = proto_map.get(alert.protocol, str(alert.protocol))

        # Build Slack message payload
        payload = {
            "channel": self.slack_config.channel,
            "username": "NIDS Alert Bot",
            "icon_emoji": ":shield:",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} NIDS Alert: {alert.attack_type} Detected"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Severity:*\n{alert.severity.upper()}"},
                        {"type": "mrkdwn", "text": f"*Confidence:*\n{alert.confidence:.1%}"},
                        {"type": "mrkdwn", "text": f"*Source:*\n`{alert.source_ip}:{alert.source_port}`"},
                        {"type": "mrkdwn", "text": f"*Destination:*\n`{alert.destination_ip}:{alert.destination_port}`"},
                        {"type": "mrkdwn", "text": f"*Protocol:*\n{proto_name}"},
                        {"type": "mrkdwn", "text": f"*Time:*\n{alert.timestamp.strftime('%H:%M:%S')}"},
                    ]
                },
                {"type": "divider"},
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Flow ID: `{alert.flow_id or 'N/A'}` | NIDS - Network Intrusion Detection System"
                        }
                    ]
                }
            ]
        }

        resp = http_requests.post(
            self.slack_config.webhook_url,
            json=payload,
            timeout=5
        )
        resp.raise_for_status()

    def _get_subject(self, alert: AttackAlert) -> str:
        """Generate email subject line."""
        severity_emoji = {
            "low": "",
            "medium": "",
            "high": "",
            "critical": ""
        }
        emoji = severity_emoji.get(alert.severity, "")
        return f"{emoji} NIDS Alert: {alert.attack_type} detected from {alert.source_ip}"

    def _get_plain_text(self, alert: AttackAlert) -> str:
        """Generate plain text email content."""
        return f"""
NETWORK INTRUSION DETECTION ALERT
==================================

Attack Type: {alert.attack_type}
Severity: {alert.severity.upper()}
Confidence: {alert.confidence:.1%}
Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Source: {alert.source_ip}:{alert.source_port}
Destination: {alert.destination_ip}:{alert.destination_port}
Protocol: {alert.protocol}
Flow ID: {alert.flow_id or 'N/A'}

Please investigate this traffic immediately.

--
NIDS - Network Intrusion Detection System
        """

    def _get_html_content(self, alert: AttackAlert) -> str:
        """Generate HTML email content."""
        severity_color = {
            "low": "#28a745",
            "medium": "#ffc107",
            "high": "#fd7e14",
            "critical": "#dc3545"
        }

        color = severity_color.get(alert.severity, "#6c757d")

        # Top features for context
        top_features_html = ""
        if alert.features:
            sorted_features = sorted(
                alert.features.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            for name, value in sorted_features:
                top_features_html += f"<tr><td>{name}</td><td>{value:.2f}</td></tr>"

        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: {color}; color: white; padding: 20px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header p {{ margin: 10px 0 0; opacity: 0.9; }}
        .content {{ padding: 20px; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 12px; text-transform: uppercase; background: {color}; color: white; }}
        .info-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        .info-table th {{ text-align: left; padding: 10px; background: #f8f9fa; border-bottom: 2px solid #dee2e6; }}
        .info-table td {{ padding: 10px; border-bottom: 1px solid #dee2e6; }}
        .alert-box {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 15px; margin: 15px 0; }}
        .footer {{ background: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #6c757d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>NIDS Alert</h1>
            <p>{alert.attack_type} Attack Detected</p>
        </div>
        <div class="content">
            <p><span class="badge">{alert.severity.upper()}</span> Confidence: {alert.confidence:.1%}</p>

            <div class="alert-box">
                <strong>Immediate Action Required</strong><br>
                A potential {alert.attack_type} attack has been detected. Please investigate the traffic below.
            </div>

            <table class="info-table">
                <tr><th colspan="2">Attack Details</th></tr>
                <tr><td>Attack Type</td><td><strong>{alert.attack_type}</strong></td></tr>
                <tr><td>Severity</td><td>{alert.severity.upper()}</td></tr>
                <tr><td>Confidence</td><td>{alert.confidence:.1%}</td></tr>
                <tr><td>Timestamp</td><td>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            </table>

            <table class="info-table">
                <tr><th colspan="2">Network Information</th></tr>
                <tr><td>Source IP</td><td><code>{alert.source_ip}:{alert.source_port}</code></td></tr>
                <tr><td>Destination IP</td><td><code>{alert.destination_ip}:{alert.destination_port}</code></td></tr>
                <tr><td>Protocol</td><td>{alert.protocol}</td></tr>
                <tr><td>Flow ID</td><td><code>{alert.flow_id or 'N/A'}</code></td></tr>
            </table>

            {f'''<table class="info-table">
                <tr><th colspan="2">Top Features</th></tr>
                {top_features_html}
            </table>''' if top_features_html else ''}
        </div>
        <div class="footer">
            NIDS - Network Intrusion Detection System<br>
            This is an automated alert. Do not reply to this email.
        </div>
    </div>
</body>
</html>
        """

    def get_statistics(self) -> dict[str, Any]:
        """
        Get alerting statistics.

        Returns:
            Dictionary with alert statistics
        """
        return {
            "email_enabled": self.email_config.enabled,
            "slack_enabled": self.slack_config.enabled,
            "total_alerts_sent": self._total_alerts_sent,
            "alerts_by_type": dict(self._alert_counts),
            "email_recipients": len(self.email_config.recipients),
            "slack_channel": self.slack_config.channel if self.slack_config.enabled else None,
        }


# ==============================================================================
# STANDALONE TESTING
# ==============================================================================

if __name__ == "__main__":
    # Test configuration
    test_config = {
        "alerting": {
            "email": {
                "enabled": False,  # Set to True and configure to test
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "use_tls": True,
                "sender": "test@example.com",
                "password": "${NIDS_EMAIL_PASSWORD}",
                "recipients": ["admin@example.com"],
                "min_confidence": 0.9,
                "cooldown_minutes": 5,
                "severity_threshold": "medium"
            }
        }
    }

    print("NIDS Alerting Module Test")
    print("=" * 50)

    manager = AlertManager(test_config)
    print(f"Alerting enabled: {manager.config.enabled}")
    print(f"Statistics: {manager.get_statistics()}")

    # Test should_send_alert logic
    print("\nTesting alert logic:")
    print(f"  DDoS at 95% confidence: {manager.should_send_alert('DDoS', 0.95)}")
    print(f"  PortScan at 95% confidence: {manager.should_send_alert('PortScan', 0.95)}")
    print(f"  DDoS at 50% confidence: {manager.should_send_alert('DDoS', 0.50)}")
